//! Monte Carlo Tree Search implementation for AlphaZero
//!
//! Optimized for practical speed:
//! - `Vec`-based children (no hashmap lookup per edge)
//! - optional time budget / early stop
//! - single-move short-circuit
//! - evaluation cache

use crate::ai::alphazero::model::AlphaZeroModel;
use crate::ai::alphazero::tensor_utils::game_to_tensor_for;
use crate::game::{BOARD_SIZE, Cell, Game, OthelloGame, Player};
use burn::tensor::activation::softmax;
use burn::tensor::backend::Backend;
use rand::RngExt;
use std::cell::RefCell;
use std::collections::HashMap;
use std::time::Instant;

/// MCTS configuration
#[derive(Debug, Clone, Copy)]
pub struct MctsConfig {
    /// Number of simulations to run.
    pub num_simulations: usize,
    /// Dirichlet noise alpha for exploration.
    pub dirichlet_alpha: f32,
    /// Dirichlet noise weight.
    pub dirichlet_weight: f32,
    /// Temperature for move selection.
    pub temperature: f32,
    /// C_puct constant for PUCT formula.
    pub c_puct: f32,
    /// Whether root Dirichlet noise is applied.
    pub add_root_dirichlet_noise: bool,
    /// Enable verbose progress logs while searching.
    pub enable_progress_log: bool,
    /// Max number of cached evaluations. Cache is cleared when limit is reached.
    pub eval_cache_size: usize,
    /// If only one valid move exists, skip MCTS and return immediately.
    pub skip_single_move_search: bool,
    /// Optional per-search time budget.
    pub max_search_time_ms: Option<u64>,
    /// Minimum simulations before time/ratio early stop can trigger.
    pub min_simulations_before_stop: usize,
    /// Early stop when best move visit ratio reaches this value.
    pub early_stop_visit_ratio: Option<f32>,
}

impl Default for MctsConfig {
    fn default() -> Self {
        Self {
            num_simulations: 600,
            dirichlet_alpha: 0.3,
            dirichlet_weight: 0.25,
            temperature: 1.0,
            c_puct: 1.2,
            add_root_dirichlet_noise: false,
            enable_progress_log: false,
            eval_cache_size: 50_000,
            skip_single_move_search: true,
            max_search_time_ms: Some(120),
            min_simulations_before_stop: 96,
            early_stop_visit_ratio: Some(0.92),
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

#[derive(Debug, Clone)]
pub struct MctsChild {
    move_idx: u8,
    node: MctsNode,
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
    /// Child nodes
    pub children: Vec<MctsChild>,
}

impl MctsNode {
    pub fn new(prior: f32) -> Self {
        Self {
            visit_count: 0,
            total_value: 0.0,
            prior,
            is_terminal: false,
            terminal_value: None,
            children: Vec::new(),
        }
    }

    pub fn new_terminal(value: f32) -> Self {
        Self {
            visit_count: 0,
            total_value: value,
            prior: 0.0,
            is_terminal: true,
            terminal_value: Some(value),
            children: Vec::new(),
        }
    }

    /// Get Q-value (average value) from this node player's perspective.
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

    /// Generic search entrypoint (works for any Game trait object).
    pub fn search(&self, game: &dyn Game) -> Vec<(usize, usize, f32)> {
        self.search_dyn(game)
    }

    /// Specialized search for OthelloGame to avoid clone_box overhead.
    pub fn search_othello(&self, game: &OthelloGame) -> Vec<(usize, usize, f32)> {
        self.search_typed(game)
    }

    fn search_dyn(&self, game: &dyn Game) -> Vec<(usize, usize, f32)> {
        let mut root = self.evaluate_node(game);
        if root.children.is_empty() {
            return Vec::new();
        }

        if self.config.skip_single_move_search && root.children.len() == 1 {
            let idx = root.children[0].move_idx as usize;
            return vec![(idx / BOARD_SIZE, idx % BOARD_SIZE, 1.0)];
        }

        if self.config.add_root_dirichlet_noise {
            self.add_dirichlet_noise(&mut root);
        }

        let start = Instant::now();
        let mut path: Vec<usize> = Vec::with_capacity(64);

        for simulation in 0..self.config.num_simulations {
            if self.config.enable_progress_log
                && self.config.num_simulations >= 10
                && simulation % (self.config.num_simulations / 10) == 0
                && simulation > 0
            {
                println!(
                    "    MCTS progress: {}/{} simulations ({}%)",
                    simulation,
                    self.config.num_simulations,
                    (simulation * 100) / self.config.num_simulations
                );
            }

            path.clear();
            let mut game_copy = game.clone_box();

            // Selection
            let mut current = &mut root;
            while !current.children.is_empty() {
                let child_index = self.select_best_child_index(current);
                let move_idx = current.children[child_index].move_idx as usize;
                path.push(child_index);

                game_copy.make_move(move_idx / BOARD_SIZE, move_idx % BOARD_SIZE);
                current = &mut current.children[child_index].node;
            }

            // Expansion + evaluation
            let leaf_value = if game_copy.is_game_over() {
                self.get_terminal_value(game_copy.as_ref(), game_copy.current_player())
            } else {
                let expanded = self.evaluate_node(game_copy.as_ref());
                let value = expanded.total_value;
                current.children = expanded.children;
                current.is_terminal = expanded.is_terminal;
                current.terminal_value = expanded.terminal_value;
                value
            };

            self.backup(&mut root, &path, leaf_value);

            if self.should_stop_early(&root, simulation + 1, start) {
                break;
            }
        }

        self.get_move_probabilities(&root)
    }

    fn search_typed<G>(&self, game: &G) -> Vec<(usize, usize, f32)>
    where
        G: Game + Clone,
    {
        let mut root = self.evaluate_node(game);
        if root.children.is_empty() {
            return Vec::new();
        }

        if self.config.skip_single_move_search && root.children.len() == 1 {
            let idx = root.children[0].move_idx as usize;
            return vec![(idx / BOARD_SIZE, idx % BOARD_SIZE, 1.0)];
        }

        if self.config.add_root_dirichlet_noise {
            self.add_dirichlet_noise(&mut root);
        }

        let start = Instant::now();
        let mut path: Vec<usize> = Vec::with_capacity(64);

        for simulation in 0..self.config.num_simulations {
            if self.config.enable_progress_log
                && self.config.num_simulations >= 10
                && simulation % (self.config.num_simulations / 10) == 0
                && simulation > 0
            {
                println!(
                    "    MCTS progress: {}/{} simulations ({}%)",
                    simulation,
                    self.config.num_simulations,
                    (simulation * 100) / self.config.num_simulations
                );
            }

            path.clear();
            let mut game_copy = game.clone();

            // Selection
            let mut current = &mut root;
            while !current.children.is_empty() {
                let child_index = self.select_best_child_index(current);
                let move_idx = current.children[child_index].move_idx as usize;
                path.push(child_index);

                game_copy.make_move(move_idx / BOARD_SIZE, move_idx % BOARD_SIZE);
                current = &mut current.children[child_index].node;
            }

            // Expansion + evaluation
            let leaf_value = if game_copy.is_game_over() {
                self.get_terminal_value(&game_copy, game_copy.current_player())
            } else {
                let expanded = self.evaluate_node(&game_copy);
                let value = expanded.total_value;
                current.children = expanded.children;
                current.is_terminal = expanded.is_terminal;
                current.terminal_value = expanded.terminal_value;
                value
            };

            self.backup(&mut root, &path, leaf_value);

            if self.should_stop_early(&root, simulation + 1, start) {
                break;
            }
        }

        self.get_move_probabilities(&root)
    }

    /// Select best child using PUCT from parent perspective.
    fn select_best_child_index(&self, node: &MctsNode) -> usize {
        let mut best_index = 0usize;
        let mut best_score = f32::MIN;
        let parent_sqrt = (node.visit_count.max(1) as f32).sqrt();

        for (index, child) in node.children.iter().enumerate() {
            // child.q_value is from child perspective (opponent), so negate.
            let q = -child.node.q_value();
            let u = self.config.c_puct * child.node.prior * parent_sqrt
                / (1.0 + child.node.visit_count as f32);
            let score = q + u;

            if score > best_score {
                best_score = score;
                best_index = index;
            }
        }

        best_index
    }

    /// Evaluate a node using neural network
    fn evaluate_node<G: Game + ?Sized>(&self, game: &G) -> MctsNode {
        if game.is_game_over() {
            let value = self.get_terminal_value(game, game.current_player());
            return MctsNode::new_terminal(value);
        }

        let key = Self::board_key(game);
        let cached = self.eval_cache.borrow().get(&key).cloned();

        let (policy_probs, value) = if let Some(cached) = cached {
            (cached.policy_probs, cached.value)
        } else {
            let input = game_to_tensor_for(game, &self.device);
            let (policy_logits, value_tensor) = self.model.forward(input);

            let policy_probs = self.logits_to_probs(&policy_logits);
            let value_data = value_tensor.to_data();
            let value_slice = value_data.as_slice::<f32>().unwrap();
            let value = value_slice.first().copied().unwrap_or(0.0);

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
        let mut children = Vec::with_capacity(valid_moves.len());
        for move_idx in valid_moves {
            children.push(MctsChild {
                move_idx,
                node: MctsNode::new(policy_probs[move_idx as usize]),
            });
        }

        let mut node = MctsNode::new(0.0);
        node.total_value = value;
        node.children = children;
        node
    }

    /// Backup values up the tree.
    fn backup(&self, root: &mut MctsNode, path: &[usize], leaf_value: f32) {
        let mut value = if path.len().is_multiple_of(2) {
            leaf_value
        } else {
            -leaf_value
        };

        root.visit_count += 1;
        root.total_value += value;

        let mut current = root;
        for &child_index in path {
            value = -value;
            current = &mut current.children[child_index].node;
            current.visit_count += 1;
            current.total_value += value;
        }
    }

    fn should_stop_early(&self, root: &MctsNode, simulations_done: usize, start: Instant) -> bool {
        if simulations_done < self.config.min_simulations_before_stop {
            return false;
        }

        if let Some(limit_ms) = self.config.max_search_time_ms
            && start.elapsed().as_millis() >= u128::from(limit_ms)
        {
            return true;
        }

        if let Some(ratio) = self.config.early_stop_visit_ratio
            && simulations_done.is_multiple_of(32)
        {
            let total_visits: usize = root.children.iter().map(|c| c.node.visit_count).sum();
            if total_visits > 0 {
                let best_visits = root
                    .children
                    .iter()
                    .map(|c| c.node.visit_count)
                    .max()
                    .unwrap_or(0);
                if best_visits as f32 / total_visits as f32 >= ratio {
                    return true;
                }
            }
        }

        false
    }

    /// Get move probabilities from visit counts
    fn get_move_probabilities(&self, root: &MctsNode) -> Vec<(usize, usize, f32)> {
        let total_visits: usize = root.children.iter().map(|c| c.node.visit_count).sum();

        let mut moves: Vec<(usize, usize, f32)> = if total_visits > 0 {
            root.children
                .iter()
                .map(|child| {
                    let idx = child.move_idx as usize;
                    (
                        idx / BOARD_SIZE,
                        idx % BOARD_SIZE,
                        child.node.visit_count as f32 / total_visits as f32,
                    )
                })
                .collect()
        } else {
            // Fallback to priors when no visit happened.
            let prior_sum: f32 = root.children.iter().map(|c| c.node.prior.max(0.0)).sum();
            if prior_sum > 0.0 {
                root.children
                    .iter()
                    .map(|child| {
                        let idx = child.move_idx as usize;
                        (
                            idx / BOARD_SIZE,
                            idx % BOARD_SIZE,
                            child.node.prior.max(0.0) / prior_sum,
                        )
                    })
                    .collect()
            } else {
                let uniform = 1.0 / root.children.len() as f32;
                root.children
                    .iter()
                    .map(|child| {
                        let idx = child.move_idx as usize;
                        (idx / BOARD_SIZE, idx % BOARD_SIZE, uniform)
                    })
                    .collect()
            }
        };

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
        let mut adjusted: Vec<(usize, usize, f32)> = moves
            .into_iter()
            .map(|(r, c, p)| (r, c, (p / max_prob).powf(1.0 / temperature) * max_prob))
            .collect();

        let sum: f32 = adjusted.iter().map(|(_, _, p)| p).sum();
        if sum > 0.0 {
            for (_, _, p) in &mut adjusted {
                *p /= sum;
            }
        }

        adjusted
    }

    /// Get valid move indices
    fn get_valid_move_indices<G: Game + ?Sized>(&self, game: &G) -> Vec<u8> {
        let mut moves = Vec::new();
        for row in 0..BOARD_SIZE {
            for col in 0..BOARD_SIZE {
                if game.is_valid_move(row, col) {
                    moves.push((row * BOARD_SIZE + col) as u8);
                }
            }
        }
        moves
    }

    /// Convert logits to probabilities using softmax
    fn logits_to_probs(&self, logits: &burn::tensor::Tensor<B, 2>) -> Vec<f32> {
        let probs = softmax(logits.clone(), 1);
        let probs_data = probs.to_data();
        probs_data.as_slice::<f32>().unwrap().to_vec()
    }

    /// Get terminal value
    fn get_terminal_value<G: Game + ?Sized>(&self, game: &G, perspective: Player) -> f32 {
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

        for (i, child) in root.children.iter_mut().enumerate() {
            child.node.prior = (1.0 - epsilon) * child.node.prior + epsilon * noise[i];
        }
    }

    fn board_key<G: Game + ?Sized>(game: &G) -> BoardKey {
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

    let threshold = rng.random::<f32>() * total;
    let mut cumulative = 0.0;

    for (row, col, prob) in move_probs {
        cumulative += prob;
        if cumulative >= threshold {
            return Some((*row, *col));
        }
    }

    move_probs.last().map(|(r, c, _)| (*r, *c))
}
