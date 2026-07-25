//! Self-play for AlphaZero training data generation
//!
//! Generates training examples by having the neural network play against itself
//! and against conventional AI opponents (Easy/Medium/SlightlyHard).

use crate::ai::alphazero::artistic::{ArtisticPattern, evaluate_artistic_reward};
use crate::ai::alphazero::mcts::{MctsConfig, MctsSearch, select_move};
use crate::ai::alphazero::tensor_utils::game_to_tensor;
use crate::ai::{AiDifficulty, AiPlayer, create_ai};
use crate::game::{BOARD_SIZE, Game, OthelloGame, Player};
use burn::tensor::backend::Backend;

/// 報酬モード: 通常 (勝敗) または芸術 (翻弄用)
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RewardMode {
    /// 通常の勝敗報酬 (+1 / 0 / -1)
    Standard,
    /// 芸術的報酬 (最終局面の美しさを評価)
    Artistic,
}

/// Configuration for self-play
#[derive(Debug, Clone, Copy)]
pub struct SelfPlayConfig {
    /// Number of self-play games
    pub num_games: usize,
    /// Number of games against conventional AI opponents (per difficulty)
    pub num_mixed_games: usize,
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
    /// Reward mode (standard or artistic)
    pub reward_mode: RewardMode,
}

impl Default for SelfPlayConfig {
    fn default() -> Self {
        Self {
            num_games: 100,
            num_mixed_games: 20,
            temperature: 1.0,
            temp_threshold: 30,
            temp_after_threshold: 0.0,
            checkpoint_interval: 10,
            mcts_config: MctsConfig::default(),
            augment_symmetry: true,
            max_moves_per_game: 120,
            reward_mode: RewardMode::Standard,
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

    /// Generate training data through self-play and mixed opponent games
    pub fn generate_data(&self) -> Vec<TrainingExample> {
        let mut all_examples = Vec::new();
        let start_time = std::time::Instant::now();
        let mut patterns: Vec<ArtisticPattern> = Vec::new();

        // Phase 1: Self-play games
        println!(
            "Starting self-play for {} games (reward: {:?})...",
            self.config.num_games, self.config.reward_mode
        );

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
            let (examples, pattern) = self.play_one_game();
            let game_duration = game_start.elapsed();

            if game_idx < 5 || game_idx % 10 == 0 {
                println!("  Game {} completed in {:?}", game_idx + 1, game_duration);
            }

            if let Some(p) = pattern {
                patterns.push(p);
            }
            all_examples.extend(examples);
        }

        // Phase 2: Mixed opponent games (vs Easy, Medium, SlightlyHard)
        if self.config.num_mixed_games > 0 {
            println!(
                "\nStarting mixed opponent games ({} per difficulty)...",
                self.config.num_mixed_games
            );

            let difficulties = [
                AiDifficulty::Easy,
                AiDifficulty::Medium,
                AiDifficulty::SlightlyHard,
            ];

            for &difficulty in &difficulties {
                println!(
                    "  Playing {} games vs {}...",
                    self.config.num_mixed_games,
                    difficulty.as_str()
                );

                for game_idx in 0..self.config.num_mixed_games {
                    let (examples, pattern) = self.play_vs_conventional_ai(difficulty);
                    if let Some(p) = pattern {
                        patterns.push(p);
                    }
                    all_examples.extend(examples);

                    if game_idx % 10 == 0 && game_idx > 0 {
                        println!("    {}/{} done", game_idx, self.config.num_mixed_games);
                    }
                }
            }
        }

        let total_time = start_time.elapsed();
        println!(
            "\nGenerated {} training examples from {} self-play + {} mixed games in {:?}",
            all_examples.len(),
            self.config.num_games,
            self.config.num_mixed_games * 3,
            total_time
        );

        // Print artistic pattern statistics if in artistic mode
        if self.config.reward_mode == RewardMode::Artistic && !patterns.is_empty() {
            crate::ai::alphazero::artistic::print_pattern_stats(&patterns);
        }

        all_examples
    }

    /// Play a single self-play game and return training examples + detected pattern
    fn play_one_game(&self) -> (Vec<TrainingExample>, Option<ArtisticPattern>) {
        let mut game = OthelloGame::default();
        let mut history: Vec<(Vec<f32>, Vec<f32>, Player)> = Vec::new();
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
                self.mcts.search_othello(&game)
            };
            let target_policy = self.move_probs_to_policy(&move_probs);

            history.push((board_tensor, target_policy, game.current_player()));

            if let Some((row, col)) = select_move(&move_probs, temperature) {
                game.make_move(row, col);
                move_count += 1;
            } else {
                game.pass_turn();
            }
        }

        let examples = self.build_examples_from_history(&game, &history);
        let pattern = if self.config.reward_mode == RewardMode::Artistic {
            Some(self.detect_final_pattern(&game))
        } else {
            None
        };

        (examples, pattern)
    }

    /// Play a game against a conventional AI opponent.
    /// AlphaZero plays as Black (first player), opponent as White.
    fn play_vs_conventional_ai(
        &self,
        difficulty: AiDifficulty,
    ) -> (Vec<TrainingExample>, Option<ArtisticPattern>) {
        let mut game = OthelloGame::default();
        let mut history: Vec<(Vec<f32>, Vec<f32>, Player)> = Vec::new();
        let mut move_count = 0usize;

        let opponent: Box<dyn AiPlayer> = create_ai(difficulty, Player::White);

        while !game.is_game_over() && move_count < self.config.max_moves_per_game {
            let valid_moves = self.valid_moves(&game);

            if valid_moves.is_empty() {
                game.pass_turn();
                continue;
            }

            if game.current_player() == Player::Black {
                // AlphaZero's turn (Black)
                let temperature = if move_count < self.config.temp_threshold {
                    self.config.temperature
                } else {
                    self.config.temp_after_threshold
                };

                let board_tensor = self.board_to_tensor(&game);

                let move_probs = if valid_moves.len() == 1 {
                    let (row, col) = valid_moves[0];
                    vec![(row, col, 1.0)]
                } else {
                    self.mcts.search_othello(&game)
                };
                let target_policy = self.move_probs_to_policy(&move_probs);

                history.push((board_tensor, target_policy, Player::Black));

                if let Some((row, col)) = select_move(&move_probs, temperature) {
                    game.make_move(row, col);
                    move_count += 1;
                } else {
                    game.pass_turn();
                }
            } else {
                // Opponent's turn (White)
                if let Some((row, col)) = opponent.choose_move(&game) {
                    game.make_move(row, col);
                    move_count += 1;
                } else {
                    game.pass_turn();
                }
            }
        }

        // Only collect training data for AlphaZero's moves (Black)
        let examples = self.build_examples_from_history(&game, &history);
        let pattern = if self.config.reward_mode == RewardMode::Artistic {
            Some(self.detect_final_pattern(&game))
        } else {
            None
        };

        (examples, pattern)
    }

    /// Build training examples from move history, applying the configured reward mode
    fn build_examples_from_history(
        &self,
        game: &OthelloGame,
        history: &[(Vec<f32>, Vec<f32>, Player)],
    ) -> Vec<TrainingExample> {
        let mut examples: Vec<TrainingExample> = Vec::with_capacity(history.len());

        for (board_tensor, target_policy, player) in history {
            let value = match self.config.reward_mode {
                RewardMode::Standard => {
                    let winner = game.winner();
                    match winner {
                        Some(w) if w == *player => 1.0,
                        Some(_) => -1.0,
                        None => 0.0,
                    }
                }
                RewardMode::Artistic => evaluate_artistic_reward(game, *player),
            };

            if self.config.augment_symmetry {
                examples.extend(self.augment_by_symmetry(board_tensor, target_policy, value));
            } else {
                examples.push(TrainingExample {
                    board_tensor: board_tensor.clone(),
                    target_policy: target_policy.clone(),
                    value,
                });
            }
        }

        examples
    }

    /// Detect the artistic pattern in the final game state
    fn detect_final_pattern(&self, game: &OthelloGame) -> ArtisticPattern {
        use crate::ai::alphazero::artistic::ArtisticPattern;

        let my_count = game.black_count();
        let opp_count = game.white_count();

        if my_count == 32 && opp_count == 32 {
            return ArtisticPattern::PerfectDraw;
        }

        // AlphaZero plays Black in mixed games; check corner sacrifice
        let corners = [(0, 0), (0, 7), (7, 0), (7, 7)];
        let opp_has_all_corners = corners
            .iter()
            .all(|&(r, c)| game.board()[r][c] == crate::game::Cell::White);
        if my_count >= 58 && opp_count <= 6 && opp_has_all_corners {
            return ArtisticPattern::CornerSacrifice;
        }

        let coverage = self.count_block_coverage(game);
        if coverage >= 0.75 {
            return ArtisticPattern::BlockTiling;
        }

        match game.winner() {
            Some(Player::Black) => {
                if my_count >= 50 {
                    ArtisticPattern::DominantWin
                } else {
                    ArtisticPattern::StandardWin
                }
            }
            Some(Player::White) => ArtisticPattern::Loss,
            None => ArtisticPattern::StandardDraw,
        }
    }

    /// Count 2x2 block coverage for pattern detection
    fn count_block_coverage(&self, game: &OthelloGame) -> f32 {
        let board = game.board();
        let mut covered = vec![vec![false; BOARD_SIZE]; BOARD_SIZE];

        for row in (0..BOARD_SIZE - 1).step_by(2) {
            for col in (0..BOARD_SIZE - 1).step_by(2) {
                let cell = board[row][col];
                if cell == crate::game::Cell::Empty {
                    continue;
                }
                if board[row][col + 1] == cell
                    && board[row + 1][col] == cell
                    && board[row + 1][col + 1] == cell
                {
                    covered[row][col] = true;
                    covered[row][col + 1] = true;
                    covered[row + 1][col] = true;
                    covered[row + 1][col + 1] = true;
                }
            }
        }

        let covered_count = covered
            .iter()
            .flat_map(|row| row.iter())
            .filter(|&&c| c)
            .count();

        covered_count as f32 / (BOARD_SIZE * BOARD_SIZE) as f32
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
