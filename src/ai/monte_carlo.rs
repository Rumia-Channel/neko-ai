//! Monte Carlo Tree Search based AI for Othello
//!
//! This AI uses Monte Carlo simulation to evaluate moves by:
//! 1. For each valid move, simulate random playouts to depth 3
//! 2. Evaluate the resulting board position by stone count
//! 3. Select the move with the best average outcome

use crate::ai::{AiPlayer, evaluate_corner_penalty};
use crate::game::{BOARD_SIZE, Game, Player};
use rand::seq::IteratorRandom;

/// Monte Carlo AI that looks ahead N moves
#[derive(Debug)]
pub struct MonteCarloAi {
    player: Player,
    /// Number of simulations per move evaluation
    simulations: usize,
    /// Maximum depth to simulate
    max_depth: usize,
}

impl MonteCarloAi {
    pub fn new(player: Player) -> Self {
        Self {
            player,
            simulations: 300, // Number of random playouts per move
            max_depth: 3,     // Look ahead 3 moves
        }
    }

    /// Create AI with custom depth and simulations
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
            // Create a copy of the game state and make the move
            let mut sim_game = game.clone_box();
            sim_game.make_move(row, col);

            // Simulate random play for remaining depth
            let score = self.simulate_playout(sim_game.as_mut(), 1);
            total_score += score;
        }

        total_score / self.simulations as f32
    }

    /// Simulate random play from current state to max_depth
    /// Returns a score based on final stone count
    fn simulate_playout(&self, game: &mut dyn Game, depth: usize) -> f32 {
        if depth >= self.max_depth || game.is_game_over() {
            return self.evaluate_board(game);
        }

        // Get all valid moves for current player
        let valid_moves: Vec<(usize, usize)> = (0..BOARD_SIZE)
            .flat_map(|r| (0..BOARD_SIZE).map(move |c| (r, c)))
            .filter(|(r, c)| game.is_valid_move(*r, *c))
            .collect();

        if valid_moves.is_empty() {
            // No valid moves, skip turn
            return self.simulate_playout(game, depth + 1);
        }

        // Choose a random move
        let mut rng = rand::rng();
        if let Some((row, col)) = valid_moves.into_iter().choose(&mut rng) {
            game.make_move(row, col);
            self.simulate_playout(game, depth + 1)
        } else {
            self.evaluate_board(game)
        }
    }

    /// Evaluate board position based on stone count
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

        // Simple evaluation: difference in stone count
        // Weighted by game phase (early game values position more)
        let total_stones = my_count + opponent_count;
        let score = if total_stones == 0 {
            0.0
        } else {
            (my_count as f32 - opponent_count as f32) / total_stones as f32
        };

        // 四隅が取られると大幅減点
        let corner_penalty = evaluate_corner_penalty(game, self.player);
        let adjusted_score = score + corner_penalty;

        // Normalize to 0-1 range for easier comparison
        (adjusted_score + 1.0) / 2.0
    }
}

impl AiPlayer for MonteCarloAi {
    fn name(&self) -> &'static str {
        "モンテカルロAI"
    }

    fn choose_move(&self, game: &dyn Game) -> Option<(usize, usize)> {
        // Get all valid moves
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

        // Evaluate each move with Monte Carlo simulation
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
