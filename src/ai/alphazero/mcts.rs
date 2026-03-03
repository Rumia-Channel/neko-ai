//! Complete Monte Carlo Tree Search implementation for AlphaZero
//!
//! Features:
//! - Tree structure with proper node management
//! - PUCT based child selection
//! - Optional Dirichlet noise for exploration
//! - Evaluation cache to avoid duplicated NN inference

use crate::ai::alphazero::model::AlphaZeroModel;
use crate::ai::alphazero::tensor_utils::game_to_tensor;
use crate::game::{BOARD_SIZE, Cell, Game, Player};
use burn::tensor::activation::softmax;
use burn::tensor::backend::Backend;
use rand::RngExt;
use std::cell::RefCell;
use std::collections::HashMap;

/// MCTS configuration
#[derive(Debug, Clone, Copy)]
pub struct MctsConfig {
    /// Number of simulations to run (default: 800)
    pub num_simulations: usize,
    /// Dirichlet noise alpha for exploration (default: 0.3)
    pub dirichlet_alpha: f32,
    /// Dirichlet noise weight (default: 0.25)
    pub dirichlet_weight: f32,
    /// Temperature for move selection (default: 1.0, 0.0 for greedy)
    pub temperature: f32,
    /// C_puct constant for UCB formula (default: 1.0)
    pub c_puct: f32,
    /// Whether root Dirichlet noise is applied.
    pub add_root_dirichlet_noise: bool,
    /// Enable verbose progress logs while searching.
    pub enable_progress_log: bool,
    /// Max number of cached evaluations. Cache is cleared when limit is reached.
    pub eval_cache_size: usize,
}

impl Default for MctsConfig {
    fn default() -> Self {
        Self {
            num_simulations: 800,
            dirichlet_alpha: 0.3,
            dirichlet_weight: 0.25,
            temperature: 1.0,
            c_puct: 1.0,
            add_root_dirichlet_noise: false,
            enable_progress_log: false,
            eval_cache_size: 50_000,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct BoardKey {
    black: u64,
    white: u64,
    current_player: Player,
}

#[derive(Debug, Clone)]
struct CachedEvaluation {
    policy_probs: Vec<f32>,
    value: f32,
}

/// Tree node for MCTS
#[derive(Debug, Clone)]
pub struct MctsNode {
    /// Number of times this node has been visited
    pub visit_count: usize,
    /// Total value accumulated from all simulations
    pub total_value: f32,
    /// Prior probability from neural network policy
    pub prior: f32,
    /// Whether this is a terminal node
    pub is_terminal: bool,
    /// Terminal value if game is over
    pub terminal_value: Option<f32>,
    /// Child nodes keyed by move index
    pub children: HashMap<usize, MctsNode>,
}

impl MctsNode {
    pub fn new(prior: f32) -> Self {
        Self {
            visit_count: 0,
            total_value: 0.0,
            prior,
            is_terminal: false,
            terminal_value: None,
            children: HashMap::new(),
        }
    }

    pub fn new_terminal(value: f32) -> Self {
        Self {
            visit_count: 0,
            total_value: value,
            prior: 0.0,
            is_terminal: true,
            terminal_value: Some(value),
            children: HashMap::new(),
        }
    }

    /// Get Q-value (average value) from the perspective of this node's current player.
    pub fn q_value(&self) -> f32 {
        if self.visit_count == 0 {
            0.0
        } else {
            self.total_value / self.visit_count as f32
        }
    }
}

/// MCTS search engine
#[derive(Debug)]
pub struct MctsSearch<B: Backend> {
    config: MctsConfig,
    model: AlphaZeroModel<B>,
    device: B::Device,
    eval_cache: RefCell<HashMap<BoardKey, CachedEvaluation>>,
}

impl<B: Backend> MctsSearch<B> {
    pub fn new(config: MctsConfig, model: AlphaZeroModel<B>, device: B::Device) -> Self {
        Self {
            config,
            model,
            device,
            eval_cache: RefCell::new(HashMap::new()),
        }
    }

    /// Perform MCTS search and return move probabilities
    pub fn search(&self, game: &dyn Game) -> Vec<(usize, usize, f32)> {
        let mut root = self.evaluate_node(game);

        if root.children.is_empty() {
            return Vec::new();
        }

        // Add Dirichlet noise to root for exploration (self-play).
        if self.config.add_root_dirichlet_noise {
            self.add_dirichlet_noise(&mut root);
        }

        let num_simulations = self.config.num_simulations;
        for i in 0..num_simulations {
            if self.config.enable_progress_log
                && num_simulations >= 10
                && i % (num_simulations / 10) == 0
                && i > 0
            {
                println!(
                    "    MCTS progress: {}/{} simulations ({}%)",
                    i,
                    num_simulations,
                    (i * 100) / num_simulations
                );
            }

            let mut game_copy = game.clone_box();
            let mut path: Vec<usize> = Vec::new();

            // Selection: traverse tree to leaf.
            let mut current = &mut root;
            while !current.children.is_empty() {
                let parent_visits = current.visit_count;
                let best_move = self.select_best_child(current, parent_visits);

                path.push(best_move);
                game_copy.make_move(best_move / BOARD_SIZE, best_move % BOARD_SIZE);

                current = current
                    .children
                    .get_mut(&best_move)
                    .expect("Child node should exist");
            }

            // Expansion + evaluation.
            let value_leaf = if game_copy.is_game_over() {
                self.get_terminal_value(game_copy.as_ref(), game_copy.current_player())
            } else {
                let new_node = self.evaluate_node(game_copy.as_ref());
                let value = new_node.total_value;
                current.children = new_node.children;
                current.is_terminal = new_node.is_terminal;
                current.terminal_value = new_node.terminal_value;
                value
            };

            self.backup(&mut root, &path, value_leaf);
        }

        self.get_move_probabilities(&root)
    }

    /// Select best child using PUCT from the parent's perspective.
    fn select_best_child(&self, node: &MctsNode, parent_visits: usize) -> usize {
        let mut best_move = 0;
        let mut best_score = f32::MIN;
        let parent_sqrt = (parent_visits.max(1) as f32).sqrt();

        for (move_idx, child) in &node.children {
            // Child q_value is from the child perspective (opponent),
            // so negate it to score from parent perspective.
            let q = -child.q_value();
            let u =
                self.config.c_puct * child.prior * parent_sqrt / (1.0 + child.visit_count as f32);
            let score = q + u;

            if score > best_score {
                best_score = score;
                best_move = *move_idx;
            }
        }

        best_move
    }

    /// Evaluate a node using neural network and return expanded priors + leaf value.
    fn evaluate_node(&self, game: &dyn Game) -> MctsNode {
        if game.is_game_over() {
            let value = self.get_terminal_value(game, game.current_player());
            return MctsNode::new_terminal(value);
        }

        let key = Self::board_key(game);
        let cached = self.eval_cache.borrow().get(&key).cloned();

        let (policy_probs, value) = if let Some(cached) = cached {
            (cached.policy_probs, cached.value)
        } else {
            let input = game_to_tensor(game, &self.device);
            let (policy_logits, value_tensor) = self.model.forward(input);

            let policy_probs = self.logits_to_probs(&policy_logits);
            let value_data = value_tensor.to_data();
            let value_slice = value_data.as_slice::<f32>().unwrap();
            let value = if value_slice.is_empty() {
                0.0
            } else {
                value_slice[0]
            };

            let mut cache = self.eval_cache.borrow_mut();
            if cache.len() >= self.config.eval_cache_size {
                cache.clear();
            }
            cache.insert(
                key,
                CachedEvaluation {
                    policy_probs: policy_probs.clone(),
                    value,
                },
            );

            (policy_probs, value)
        };

        let valid_moves = self.get_valid_move_indices(game);
        let mut children = HashMap::with_capacity(valid_moves.len());
        for move_idx in valid_moves {
            let prior = policy_probs[move_idx];
            children.insert(move_idx, MctsNode::new(prior));
        }

        let mut node = MctsNode::new(0.0);
        node.total_value = value;
        node.children = children;
        node
    }

    /// Backup values up the tree.
    /// `value_leaf` is from the perspective of the leaf node current player.
    fn backup(&self, root: &mut MctsNode, path: &[usize], value_leaf: f32) {
        // Convert leaf value to root perspective.
        let mut value = if path.len().is_multiple_of(2) {
            value_leaf
        } else {
            -value_leaf
        };

        root.visit_count += 1;
        root.total_value += value;

        let mut current = root;
        for move_idx in path {
            // Move to child: perspective flips.
            value = -value;
            current = current
                .children
                .get_mut(move_idx)
                .expect("Child node should exist");
            current.visit_count += 1;
            current.total_value += value;
        }
    }

    /// Get move probabilities from visit counts
    fn get_move_probabilities(&self, root: &MctsNode) -> Vec<(usize, usize, f32)> {
        let total_visits: usize = root.children.values().map(|c| c.visit_count).sum();

        if total_visits == 0 {
            return Vec::new();
        }

        let mut moves: Vec<(usize, usize, f32)> = root
            .children
            .iter()
            .map(|(move_idx, child)| {
                let row = move_idx / BOARD_SIZE;
                let col = move_idx % BOARD_SIZE;
                let prob = child.visit_count as f32 / total_visits as f32;
                (row, col, prob)
            })
            .collect();

        if self.config.temperature != 1.0 && self.config.temperature > 0.0 {
            moves = self.apply_temperature(moves);
        }

        moves
    }

    /// Apply temperature to move probabilities
    fn apply_temperature(&self, moves: Vec<(usize, usize, f32)>) -> Vec<(usize, usize, f32)> {
        let max_prob = moves.iter().map(|(_, _, p)| *p).fold(0.0f32, f32::max);

        if max_prob <= 0.0 {
            return moves;
        }

        let temperature = self.config.temperature;
        let mut new_moves: Vec<(usize, usize, f32)> = moves
            .into_iter()
            .map(|(r, c, p)| {
                let adjusted = (p / max_prob).powf(1.0 / temperature) * max_prob;
                (r, c, adjusted)
            })
            .collect();

        let sum: f32 = new_moves.iter().map(|(_, _, p)| p).sum();
        if sum > 0.0 {
            for (_, _, p) in &mut new_moves {
                *p /= sum;
            }
        }

        new_moves
    }

    /// Get valid move indices
    fn get_valid_move_indices(&self, game: &dyn Game) -> Vec<usize> {
        let mut moves = Vec::new();
        for row in 0..BOARD_SIZE {
            for col in 0..BOARD_SIZE {
                if game.is_valid_move(row, col) {
                    moves.push(row * BOARD_SIZE + col);
                }
            }
        }
        moves
    }

    /// Convert logits to probabilities using softmax
    fn logits_to_probs(&self, logits: &burn::tensor::Tensor<B, 2>) -> Vec<f32> {
        let probs = softmax(logits.clone(), 1);
        let probs_data = probs.to_data();
        let data = probs_data.as_slice::<f32>().unwrap();
        data.to_vec()
    }

    /// Get terminal value
    fn get_terminal_value(&self, game: &dyn Game, perspective: Player) -> f32 {
        match game.winner() {
            Some(winner) if winner == perspective => 1.0,
            Some(_) => -1.0,
            None => 0.0,
        }
    }

    /// Add Dirichlet noise to root node
    fn add_dirichlet_noise(&self, root: &mut MctsNode) {
        if root.children.is_empty() {
            return;
        }

        let n = root.children.len();
        let alpha = self.config.dirichlet_alpha.max(0.05);
        let epsilon = self.config.dirichlet_weight.clamp(0.0, 1.0);

        let mut rng = rand::rng();
        let samples_per_dim = alpha.ceil() as usize;
        let mut noise = vec![0.0f32; n];
        for value in &mut noise {
            let lambda = 1.0 / alpha as f64;
            let gamma: f64 = (0..samples_per_dim)
                .map(|_| {
                    let u: f64 = rng.random::<f64>().clamp(1e-12, 1.0);
                    -u.ln() / lambda
                })
                .sum();
            *value = gamma as f32;
        }

        let sum: f32 = noise.iter().sum();
        if sum <= 0.0 {
            return;
        }
        for value in &mut noise {
            *value /= sum;
        }

        for (i, child) in root.children.values_mut().enumerate() {
            child.prior = (1.0 - epsilon) * child.prior + epsilon * noise[i];
        }
    }

    fn board_key(game: &dyn Game) -> BoardKey {
        let mut black = 0u64;
        let mut white = 0u64;

        for row in 0..BOARD_SIZE {
            for col in 0..BOARD_SIZE {
                let bit = 1u64 << (row * BOARD_SIZE + col);
                match game.board()[row][col] {
                    Cell::Black => black |= bit,
                    Cell::White => white |= bit,
                    Cell::Empty => {}
                }
            }
        }

        BoardKey {
            black,
            white,
            current_player: game.current_player(),
        }
    }

    /// Get device
    pub fn device(&self) -> &B::Device {
        &self.device
    }

    /// Clear cached NN evaluations.
    pub fn clear_eval_cache(&self) {
        self.eval_cache.borrow_mut().clear();
    }
}

/// Select move from probabilities using temperature
pub fn select_move(move_probs: &[(usize, usize, f32)], temperature: f32) -> Option<(usize, usize)> {
    if move_probs.is_empty() {
        return None;
    }

    let mut rng = rand::rng();

    if temperature <= 0.0 {
        return move_probs
            .iter()
            .max_by(|(_, _, p1), (_, _, p2)| p1.partial_cmp(p2).unwrap())
            .map(|(r, c, _)| (*r, *c));
    }

    let total: f32 = move_probs.iter().map(|(_, _, p)| p).sum();
    if total <= 0.0 {
        let idx: usize = rng.random_range(0..move_probs.len());
        let (r, c, _) = move_probs[idx];
        return Some((r, c));
    }

    let threshold: f32 = rng.random::<f32>() * total;
    let mut cumulative = 0.0;

    for (row, col, prob) in move_probs {
        cumulative += prob;
        if cumulative >= threshold {
            return Some((*row, *col));
        }
    }

    move_probs.last().map(|(r, c, _)| (*r, *c))
}
