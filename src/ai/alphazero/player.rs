//! AlphaZero AI Player
//!
//! Combines MCTS with neural network guidance to play Othello.

use crate::ai::alphazero::mcts::{MctsConfig, MctsSearch};
use crate::ai::alphazero::model::AlphaZeroModel;
use crate::ai::AiPlayer;
use crate::game::{Game, Player};
use burn::tensor::backend::Backend;

/// AlphaZero AI player that uses neural network + MCTS
#[derive(Debug)]
pub struct AlphaZeroPlayer<B: Backend> {
    player: Player,
    mcts_search: MctsSearch<B>,
}

impl<B: Backend> AlphaZeroPlayer<B> {
    /// Create a new AlphaZero player
    pub fn new(player: Player, mcts_search: MctsSearch<B>) -> Self {
        Self {
            player,
            mcts_search,
        }
    }

    /// Create player with default configuration
    pub fn with_model(player: Player, model: AlphaZeroModel<B>, device: B::Device) -> Self {
        let config = MctsConfig::default();
        let search = MctsSearch::new(config, model, device);
        Self::new(player, search)
    }

    /// Get the MCTS search instance (for training)
    pub fn mcts_search(&self) -> &MctsSearch<B> {
        &self.mcts_search
    }
}

impl<B: Backend> AiPlayer for AlphaZeroPlayer<B> {
    fn name(&self) -> &'static str {
        "AlphaZero"
    }

    fn choose_move(&self, game: &dyn Game) -> Option<(usize, usize)> {
        // Run MCTS search
        let move_probs = self.mcts_search.search(game);

        // Select move based on probabilities (greedy for gameplay)
        if move_probs.is_empty() {
            return None;
        }

        // Find move with highest probability
        let best = move_probs.into_iter().max_by(
            |(_, _, p1): &(usize, usize, f32), (_, _, p2): &(usize, usize, f32)| {
                p1.partial_cmp(p2).unwrap()
            },
        );

        best.map(|(row, col, _)| (row, col))
    }

    fn player(&self) -> Player {
        self.player
    }
}

/// Configuration for AlphaZero player
#[derive(Debug, Clone, Copy, Default)]
pub struct AlphaZeroConfig {
    pub mcts_config: MctsConfig,
}
