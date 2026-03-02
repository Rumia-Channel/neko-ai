//! 翻弄AI - 最上級難易度
//!
//! このAIは複合的な評価関数を使用して、プレイヤーを翻弄するような手を選ぶ
//! 現在はプレースホルダーとして、より深い探索と多いシミュレーション数を設定

use crate::ai::AiPlayer;
use crate::game::{BOARD_SIZE, Game, Player};
use rand::seq::IteratorRandom;

/// 翻弄AI - 最上級難易度
#[derive(Debug)]
pub struct HonrouAi {
    player: Player,
    /// Number of simulations per move evaluation
    simulations: usize,
    /// Maximum depth to simulate
    max_depth: usize,
}

impl HonrouAi {
    pub fn new(player: Player) -> Self {
        Self {
            player,
            simulations: 1000, // 多いシミュレーション回数
            max_depth: 7,      // 深い探索
        }
    }

    /// Create AI with custom depth and simulations
    #[allow(dead_code)]
    pub fn with_config(player: Player, depth: usize, simulations: usize) -> Self {
        Self {
            player,
            simulations,
            max_depth: depth,
        }
    }

    /// Evaluate a move by running Monte Carlo simulations
    fn evaluate_move(&self, game: &dyn Game, row: usize, col: usize) -> f32 {
        let mut total_score = 0.0;

        for _ in 0..self.simulations {
            let mut sim_game = game.clone_box();
            sim_game.make_move(row, col);
            let score = self.simulate_playout(sim_game.as_mut(), 1);
            total_score += score;
        }

        total_score / self.simulations as f32
    }

    /// Simulate random play from current state to max_depth
    fn simulate_playout(&self, game: &mut dyn Game, depth: usize) -> f32 {
        if depth >= self.max_depth || game.is_game_over() {
            return self.evaluate_board(game);
        }

        let valid_moves: Vec<(usize, usize)> = (0..BOARD_SIZE)
            .flat_map(|r| (0..BOARD_SIZE).map(move |c| (r, c)))
            .filter(|(r, c)| game.is_valid_move(*r, *c))
            .collect();

        if valid_moves.is_empty() {
            return self.simulate_playout(game, depth + 1);
        }

        let mut rng = rand::rng();
        if let Some((row, col)) = valid_moves.into_iter().choose(&mut rng) {
            game.make_move(row, col);
            self.simulate_playout(game, depth + 1)
        } else {
            self.evaluate_board(game)
        }
    }

    /// Evaluate board position
    /// Returns positive score if favorable for our player
    fn evaluate_board(&self, game: &dyn Game) -> f32 {
        let my_count = match self.player {
            Player::Black => game.black_count(),
            Player::White => game.white_count(),
        };
        let opponent_count = match self.player {
            Player::Black => game.white_count(),
            Player::White => game.black_count(),
        };

        let total_stones = my_count + opponent_count;
        let score = if total_stones == 0 {
            0.0
        } else {
            (my_count as f32 - opponent_count as f32) / total_stones as f32
        };

        (score + 1.0) / 2.0
    }
}

impl AiPlayer for HonrouAi {
    fn name(&self) -> &'static str {
        "翻弄AI"
    }

    fn choose_move(&self, game: &dyn Game) -> Option<(usize, usize)> {
        let valid_moves: Vec<(usize, usize)> = (0..BOARD_SIZE)
            .flat_map(|row| (0..BOARD_SIZE).map(move |col| (row, col)))
            .filter(|(row, col)| game.is_valid_move(*row, *col))
            .collect();

        if valid_moves.is_empty() {
            return None;
        }

        if valid_moves.len() == 1 {
            return Some(valid_moves[0]);
        }

        let mut best_move = valid_moves[0];
        let mut best_score = f32::MIN;

        for (row, col) in valid_moves {
            let score = self.evaluate_move(game, row, col);

            if score > best_score {
                best_score = score;
                best_move = (row, col);
            }
        }

        Some(best_move)
    }

    fn player(&self) -> Player {
        self.player
    }
}
