//! Monte Carlo Tree Search with neural network guidance
//!
//! Simplified placeholder for AlphaZero MCTS.

use crate::game::{Game, BOARD_SIZE};
use burn::tensor::backend::Backend;

/// MCTS configuration
#[derive(Debug, Clone, Copy)]
pub struct MctsConfig {
    pub num_simulations: usize,
    pub temperature: f32,
}

impl Default for MctsConfig {
    fn default() -> Self {
        Self {
            num_simulations: 800,
            temperature: 1.0,
        }
    }
}

/// MCTS search placeholder
#[derive(Debug)]
pub struct MctsSearch<B: Backend> {
    config: MctsConfig,
    device: B::Device,
}

impl<B: Backend> MctsSearch<B> {
    pub fn new(
        config: MctsConfig,
        _model: crate::ai::alphazero::model::AlphaZeroModel<B>,
        device: B::Device,
    ) -> Self {
        Self { config, device }
    }

    /// Search for the best move - placeholder implementation
    pub fn search(&self, game: &dyn Game) -> Vec<(usize, usize, f32)> {
        // Simple implementation: return valid moves with uniform probability
        let mut moves = Vec::new();
        for row in 0..BOARD_SIZE {
            for col in 0..BOARD_SIZE {
                if game.is_valid_move(row, col) {
                    moves.push((row, col, 1.0));
                }
            }
        }

        // Normalize probabilities
        if !moves.is_empty() {
            let count = moves.len() as f32;
            for (_, _, prob) in &mut moves {
                *prob = 1.0 / count;
            }
        }

        moves
    }

    /// Get the device
    pub fn device(&self) -> &B::Device {
        &self.device
    }
}
