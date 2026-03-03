//! Self-play for AlphaZero training data generation
//!
//! Generates training examples by having the neural network play against itself.

use crate::ai::alphazero::mcts::{MctsConfig, MctsSearch, select_move};
use crate::ai::alphazero::tensor_utils::game_to_tensor;
use crate::game::{BOARD_SIZE, Game, OthelloGame, Player};
use burn::tensor::backend::Backend;
use std::collections::VecDeque;

/// Configuration for self-play
#[derive(Debug, Clone, Copy)]
pub struct SelfPlayConfig {
    /// Number of games to play (default: 25000 in full AlphaZero)
    pub num_games: usize,
    /// Temperature for early game exploration (default: 1.0)
    pub temperature: f32,
    /// Number of moves before applying temperature decay (default: 30)
    pub temp_threshold: usize,
    /// Temperature after threshold (default: 0.0 for greedy)
    pub temp_after_threshold: f32,
    /// Checkpoint interval (default: 1000)
    pub checkpoint_interval: usize,
    /// MCTS configuration
    pub mcts_config: MctsConfig,
    /// Augment each example using board symmetries.
    pub augment_symmetry: bool,
    /// Safety guard for infinite loops.
    pub max_moves_per_game: usize,
}

impl Default for SelfPlayConfig {
    fn default() -> Self {
        Self {
            num_games: 100,
            temperature: 1.0,
            temp_threshold: 30,
            temp_after_threshold: 0.0,
            checkpoint_interval: 10,
            mcts_config: MctsConfig::default(),
            augment_symmetry: true,
            max_moves_per_game: 120,
        }
    }
}

/// Single training example
#[derive(Debug, Clone)]
pub struct TrainingExample {
    /// Board state as tensor (3x8x8)
    pub board_tensor: Vec<f32>,
    /// Target policy (64 probabilities)
    pub target_policy: Vec<f32>,
    /// Final game outcome (-1.0, 0.0, or 1.0)
    pub value: f32,
}

/// Self-play engine
pub struct SelfPlayEngine<B: Backend> {
    config: SelfPlayConfig,
    mcts: MctsSearch<B>,
    device: B::Device,
}

impl<B: Backend> SelfPlayEngine<B> {
    pub fn new(config: SelfPlayConfig, mcts: MctsSearch<B>, device: B::Device) -> Self {
        Self {
            config,
            mcts,
            device,
        }
    }

    /// Generate training data through self-play
    pub fn generate_data(&self) -> Vec<TrainingExample> {
        let mut all_examples = Vec::new();
        let start_time = std::time::Instant::now();

        println!("Starting self-play for {} games...", self.config.num_games);

        for game_idx in 0..self.config.num_games {
            if game_idx % self.config.checkpoint_interval == 0 {
                let elapsed = start_time.elapsed().as_secs_f32();
                let games_per_sec = if elapsed > 0.0 {
                    game_idx as f32 / elapsed
                } else {
                    0.0
                };
                println!(
                    "Self-play progress: {}/{} games ({:.1} games/sec, {:.1}s elapsed)",
                    game_idx, self.config.num_games, games_per_sec, elapsed
                );
            }

            let game_start = std::time::Instant::now();
            let examples = self.play_one_game();
            let game_duration = game_start.elapsed();

            if game_idx < 5 || game_idx % 10 == 0 {
                println!("  Game {} completed in {:?}", game_idx + 1, game_duration);
            }

            all_examples.extend(examples);
        }

        let total_time = start_time.elapsed();
        println!(
            "Generated {} training examples from {} games in {:?}",
            all_examples.len(),
            self.config.num_games,
            total_time
        );

        all_examples
    }

    /// Play a single game and return training examples
    fn play_one_game(&self) -> Vec<TrainingExample> {
        let mut game = OthelloGame::default();
        let mut history: VecDeque<(Vec<f32>, Vec<f32>, Player)> = VecDeque::new();
        let mut move_count = 0usize;

        while !game.is_game_over() && move_count < self.config.max_moves_per_game {
            let valid_moves = self.valid_moves(&game);

            if valid_moves.is_empty() {
                game.pass_turn();
                continue;
            }

            let temperature = if move_count < self.config.temp_threshold {
                self.config.temperature
            } else {
                self.config.temp_after_threshold
            };

            let board_tensor = self.board_to_tensor(&game);

            // Skip expensive MCTS when move is forced.
            let move_probs = if valid_moves.len() == 1 {
                let (row, col) = valid_moves[0];
                vec![(row, col, 1.0)]
            } else {
                self.mcts.search(&game)
            };
            let target_policy = self.move_probs_to_policy(&move_probs);

            history.push_back((board_tensor, target_policy, game.current_player()));

            if let Some((row, col)) = select_move(&move_probs, temperature) {
                game.make_move(row, col);
                move_count += 1;
            } else {
                // Should be rare: fallback to pass for safety.
                game.pass_turn();
            }
        }

        let winner = game.winner();
        let final_value = |player: Player| -> f32 {
            match winner {
                Some(w) if w == player => 1.0,
                Some(_) => -1.0,
                None => 0.0,
            }
        };

        let mut examples: Vec<TrainingExample> = Vec::with_capacity(history.len());
        for (board_tensor, target_policy, player) in history {
            let value = final_value(player);
            if self.config.augment_symmetry {
                examples.extend(self.augment_by_symmetry(&board_tensor, &target_policy, value));
            } else {
                examples.push(TrainingExample {
                    board_tensor,
                    target_policy,
                    value,
                });
            }
        }

        examples
    }

    fn valid_moves(&self, game: &OthelloGame) -> Vec<(usize, usize)> {
        (0..BOARD_SIZE)
            .flat_map(|r| (0..BOARD_SIZE).map(move |c| (r, c)))
            .filter(|(r, c)| game.is_valid_move(*r, *c))
            .collect()
    }

    /// Convert board to tensor format
    fn board_to_tensor(&self, game: &OthelloGame) -> Vec<f32> {
        let tensor = game_to_tensor::<B>(game, &self.device);
        let tensor_data = tensor.to_data();
        let data = tensor_data.as_slice::<f32>().unwrap();
        data.to_vec()
    }

    /// Convert move probabilities to policy vector
    fn move_probs_to_policy(&self, move_probs: &[(usize, usize, f32)]) -> Vec<f32> {
        let mut policy = vec![0.0f32; BOARD_SIZE * BOARD_SIZE];

        for (row, col, prob) in move_probs {
            let idx = row * BOARD_SIZE + col;
            if idx < BOARD_SIZE * BOARD_SIZE {
                policy[idx] = *prob;
            }
        }

        let sum: f32 = policy.iter().sum();
        if sum > 0.0 {
            for p in &mut policy {
                *p /= sum;
            }
        }

        policy
    }

    fn augment_by_symmetry(
        &self,
        board_tensor: &[f32],
        policy: &[f32],
        value: f32,
    ) -> Vec<TrainingExample> {
        let mut examples = Vec::with_capacity(8);
        for symmetry in 0..8 {
            let board = self.transform_board(board_tensor, symmetry);
            let policy = self.transform_policy(policy, symmetry);
            examples.push(TrainingExample {
                board_tensor: board,
                target_policy: policy,
                value,
            });
        }
        examples
    }

    fn transform_board(&self, board_tensor: &[f32], symmetry: usize) -> Vec<f32> {
        // board tensor shape is [1, 3, 8, 8], flattened.
        let mut transformed = vec![0.0f32; board_tensor.len()];
        for channel in 0..3usize {
            let channel_offset = channel * BOARD_SIZE * BOARD_SIZE;
            for row in 0..BOARD_SIZE {
                for col in 0..BOARD_SIZE {
                    let src_idx = channel_offset + row * BOARD_SIZE + col;
                    let (new_row, new_col) = Self::map_coord(row, col, symmetry);
                    let dst_idx = channel_offset + new_row * BOARD_SIZE + new_col;
                    transformed[dst_idx] = board_tensor[src_idx];
                }
            }
        }
        transformed
    }

    fn transform_policy(&self, policy: &[f32], symmetry: usize) -> Vec<f32> {
        let mut transformed = vec![0.0f32; BOARD_SIZE * BOARD_SIZE];
        for row in 0..BOARD_SIZE {
            for col in 0..BOARD_SIZE {
                let src_idx = row * BOARD_SIZE + col;
                let (new_row, new_col) = Self::map_coord(row, col, symmetry);
                let dst_idx = new_row * BOARD_SIZE + new_col;
                transformed[dst_idx] = policy[src_idx];
            }
        }
        transformed
    }

    fn map_coord(row: usize, col: usize, symmetry: usize) -> (usize, usize) {
        let last = BOARD_SIZE - 1;
        match symmetry {
            0 => (row, col),               // identity
            1 => (col, last - row),        // rot90
            2 => (last - row, last - col), // rot180
            3 => (last - col, row),        // rot270
            4 => (row, last - col),        // mirror horizontal
            5 => (last - row, col),        // mirror vertical
            6 => (col, row),               // transpose
            7 => (last - col, last - row), // anti-diagonal
            _ => (row, col),
        }
    }
}
