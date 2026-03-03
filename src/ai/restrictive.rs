//! Monte Carlo AI with opponent mobility minimization strategy
//!
//! This AI uses Monte Carlo simulation to minimize opponent's options:
//! 1. For each valid move, simulate random playouts to depth 5
//! 2. Evaluate the resulting board by counting opponent's valid moves
//! 3. Select the move that minimizes opponent's mobility

use crate::ai::AiPlayer;
use crate::game::{BOARD_SIZE, Game, Player};
use rand::seq::IteratorRandom;

/// Monte Carlo AI that minimizes opponent's mobility
#[derive(Debug)]
pub struct RestrictiveAi {
    player: Player,
    /// Number of simulations per move evaluation
    simulations: usize,
    /// Maximum depth to simulate (5 moves)
    max_depth: usize,
}

impl RestrictiveAi {
    pub fn new(player: Player) -> Self {
        Self {
            player,
            simulations: 2000, // More simulations for better evaluation
            max_depth: 5,      // Look ahead 5 moves
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
    /// Returns a score where higher is better (lower opponent mobility)
    /// Heavily penalizes moves that give opponent access to corners
    fn evaluate_move(&self, game: &dyn Game, row: usize, col: usize) -> f32 {
        // First, check if this move gives opponent a corner opportunity
        let mut immediate_game = game.clone_box();
        immediate_game.make_move(row, col);

        // Check if opponent can take any corners after this move
        let corner_penalty = self.evaluate_corner_vulnerability(&*immediate_game);

        // Run Monte Carlo simulations
        let mut total_score = 0.0;

        for _ in 0..self.simulations {
            // Create a copy of the game state after the move
            let mut sim_game = immediate_game.clone_box();

            // Simulate random play for remaining depth
            let score = self.simulate_playout(sim_game.as_mut(), 1);
            total_score += score;
        }

        let avg_score = total_score / self.simulations as f32;

        // Apply heavy penalty for giving opponent corner opportunities
        // This prioritizes avoiding corner giveaways over other factors
        avg_score + corner_penalty
    }

    /// Check if opponent can take any corners after current move
    /// Returns a large negative penalty if corners are vulnerable
    fn evaluate_corner_vulnerability(&self, game: &dyn Game) -> f32 {
        let _opponent = self.player.opposite();
        let corners = [(0, 0), (0, 7), (7, 0), (7, 7)];
        let mut penalty = 0.0f32;

        for (row, col) in corners {
            if game.is_valid_move(row, col) {
                // Opponent can take this corner - heavy penalty!
                penalty -= 2.0; // 各四隅に-2.0のペナルティ
            }
        }

        penalty
    }

    /// Simulate random play from current state to max_depth
    /// Returns a score based on opponent's mobility (fewer moves = higher score)
    fn simulate_playout(&self, game: &mut dyn Game, depth: usize) -> f32 {
        if depth >= self.max_depth || game.is_game_over() {
            return self.evaluate_opponent_mobility(game);
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
            self.evaluate_opponent_mobility(game)
        }
    }

    /// Evaluate board position based on opponent's mobility
    /// Returns higher score when opponent has fewer valid moves
    fn evaluate_opponent_mobility(&self, game: &dyn Game) -> f32 {
        let _opponent = self.player.opposite();

        // Count opponent's valid moves
        let opponent_moves: usize = (0..BOARD_SIZE)
            .flat_map(|r| (0..BOARD_SIZE).map(move |c| (r, c)))
            .filter(|(r, c)| game.is_valid_move(*r, *c))
            .count();

        // Also consider our own mobility (we want to maximize our options)
        let our_moves: usize = (0..BOARD_SIZE)
            .flat_map(|r| (0..BOARD_SIZE).map(move |c| (r, c)))
            .filter(|(r, c)| {
                // Check if this would be a valid move for us (current player in simulation)
                // Note: After simulation, the current player might be different
                // We evaluate from the perspective of the original AI player
                let current = game.current_player();
                if current == self.player {
                    game.is_valid_move(*r, *c)
                } else {
                    false
                }
            })
            .count();

        // Score: minimize opponent moves, maximize our moves
        // Opponent mobility penalty: more opponent moves = lower score
        let max_possible_moves = (BOARD_SIZE * BOARD_SIZE) as f32;
        let opponent_penalty = opponent_moves as f32 / max_possible_moves;

        // Our mobility bonus: more our moves = higher score
        let our_bonus = our_moves as f32 / max_possible_moves;

        // Combined score: higher is better
        // Weight: opponent restriction (70%) + our mobility (30%)
        (1.0 - opponent_penalty) * 0.7 + our_bonus * 0.3
    }
}

impl AiPlayer for RestrictiveAi {
    fn name(&self) -> &'static str {
        "制限的AI"
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
