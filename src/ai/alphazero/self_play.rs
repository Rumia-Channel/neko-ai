//! Self-play for AlphaZero training data generation
//!
//! Generates training examples by having the neural network play against itself.

use crate::ai::alphazero::mcts::{MctsConfig, MctsSearch, select_move};
use crate::ai::alphazero::tensor_utils::game_to_tensor;
use crate::game::{Game, OthelloGame, Player};
use burn::tensor::backend::Backend;
use rand::RngExt;
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

        println!("Starting self-play for {} games...", self.config.num_games);

        for game_idx in 0..self.config.num_games {
            if game_idx % self.config.checkpoint_interval == 0 {
                println!(
                    "Self-play progress: {}/{} games",
                    game_idx, self.config.num_games
                );
            }

            let examples = self.play_one_game();
            all_examples.extend(examples);
        }

        println!(
            "Generated {} training examples from {} games",
            all_examples.len(),
            self.config.num_games
        );

        all_examples
    }

    /// Play a single game and return training examples
    fn play_one_game(&self) -> Vec<TrainingExample> {
        let mut game = OthelloGame::default();
        let mut history: VecDeque<(TrainingExample, Player)> = VecDeque::new();
        let mut move_count = 0;

        while !game.is_game_over() {
            // Get valid moves
            let valid_moves: Vec<(usize, usize)> = (0..8)
                .flat_map(|r| (0..8).map(move |c| (r, c)))
                .filter(|(r, c)| game.is_valid_move(*r, *c))
                .collect();

            if valid_moves.is_empty() {
                // Pass turn
                game.make_move(0, 0); // Invalid move triggers pass
                continue;
            }

            // Run MCTS search
            let move_probs = self.mcts.search(&game);

            // Apply temperature
            let temperature = if move_count < self.config.temp_threshold {
                self.config.temperature
            } else {
                self.config.temp_after_threshold
            };

            // Store training example
            let board_tensor = self.board_to_tensor(&game);
            let target_policy = self.move_probs_to_policy(&move_probs, valid_moves.len());

            history.push_back((
                TrainingExample {
                    board_tensor,
                    target_policy,
                    value: 0.0, // Will be filled in after game ends
                },
                game.current_player(),
            ));

            // Select move
            if let Some((row, col)) = select_move(&move_probs, temperature) {
                game.make_move(row, col);
                move_count += 1;
            }
        }

        // Determine game result
        let winner = game.winner();
        let final_value = |player: Player| -> f32 {
            match winner {
                Some(w) if w == player => 1.0,
                Some(_) => -1.0,
                None => 0.0,
            }
        };

        // Update all examples with actual game outcome
        let mut examples: Vec<TrainingExample> = Vec::with_capacity(history.len());
        for (mut example, player) in history {
            example.value = final_value(player);
            examples.push(example);
        }

        examples
    }

    /// Convert board to tensor format
    fn board_to_tensor(&self, game: &OthelloGame) -> Vec<f32> {
        let tensor = game_to_tensor::<B>(game, &self.device);
        let tensor_data = tensor.to_data();
        let data = tensor_data.as_slice::<f32>().unwrap();
        data.to_vec()
    }

    /// Convert move probabilities to policy vector
    fn move_probs_to_policy(
        &self,
        move_probs: &[(usize, usize, f32)],
        _num_valid: usize,
    ) -> Vec<f32> {
        let mut policy = vec![0.0f32; 64];

        for (row, col, prob) in move_probs {
            let idx = row * 8 + col;
            if idx < 64 {
                policy[idx] = *prob;
            }
        }

        // Normalize
        let sum: f32 = policy.iter().sum();
        if sum > 0.0 {
            for p in &mut policy {
                *p /= sum;
            }
        }

        policy
    }
}
