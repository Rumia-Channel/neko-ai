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
use burn::tensor::{Device, Tensor, TensorData};
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
    /// Number of leaves evaluated per neural-network forward pass.
    ///
    /// 既定は 1（1 リーフずつ評価）。バッチ選択は逐次選択ではないため、
    /// 2 以上にすると同じリーフを重複評価しやすく、探索が伸びないわりに
    /// 計算だけ増える（実測: batch 8 は 1 リーフあたり約 3 倍の時間）。
    pub eval_batch_size: usize,
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
            eval_batch_size: 1,
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
    /// このノードで手番を持つプレイヤー。
    ///
    /// オセロはパスがあるため 1 手（1 エッジ）で手番が入れ替わらないことがある。
    /// 値の符号はパリティではなく、この手番の一致で決める必要がある。
    /// 子ノードは降下時に確定する（未訪問ノードの Q 値は 0 なので選択には影響しない）。
    pub player: Player,
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
            // 手番は降下時に設定される
            player: Player::Black,
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
            player: Player::Black,
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

/// ノードの手番視点の値へ変換する。
///
/// `leaf_player` と手番が同じならそのまま、違えば符号を反転する。
/// オセロはパスがあるため、経路長のパリティでは正しい符号にならない。
fn orient_for(node_player: Player, leaf_player: Player, leaf_value: f32) -> f32 {
    if node_player == leaf_player {
        leaf_value
    } else {
        -leaf_value
    }
}

/// MCTS search engine
#[derive(Debug)]
pub struct MctsSearch {
    config: MctsConfig,
    model: AlphaZeroModel,
    device: Device,
    eval_cache: RefCell<HashMap<BoardKey, CachedEvaluation>>,
}

impl MctsSearch {
    pub fn new(config: MctsConfig, model: AlphaZeroModel, device: Device) -> Self {
        Self {
            config,
            model,
            device,
            eval_cache: RefCell::new(HashMap::new()),
        }
    }

    /// 並列自己対戦用に、同じモデル（重みは共有）を持つ独立した探索器を作る。
    ///
    /// 評価キャッシュは共有せず、ワーカーごとに持たせる。
    pub fn fork(&self) -> Self {
        Self {
            config: self.config,
            model: self.model.clone(),
            device: self.device.clone(),
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
                // パスがあるため、子ノードの手番は実際の局面から取得する
                current.player = game_copy.current_player();
            }

            // Expansion + evaluation
            let leaf_player = game_copy.current_player();
            let leaf_value = if game_copy.is_game_over() {
                self.get_terminal_value(game_copy.as_ref(), leaf_player)
            } else {
                let expanded = self.evaluate_node(game_copy.as_ref());
                let value = expanded.total_value;
                current.children = expanded.children;
                current.is_terminal = expanded.is_terminal;
                current.terminal_value = expanded.terminal_value;
                current.player = expanded.player;
                value
            };

            Self::backup(&mut root, &path, leaf_player, leaf_value);

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
        let batch_size = self.config.eval_batch_size.max(1);
        let mut simulation = 0usize;

        while simulation < self.config.num_simulations {
            let batch_end = (simulation + batch_size).min(self.config.num_simulations);
            let current_batch = batch_end - simulation;

            // Phase 1: Selection — collect leaves (read-only tree traversal)
            let mut leaves: Vec<(Vec<usize>, Vec<Player>, G)> = Vec::with_capacity(current_batch);
            for _ in 0..current_batch {
                let (path, players, leaf_game) = self.select_leaf_typed(&root, game);
                leaves.push((path, players, leaf_game));
            }

            // Phase 2: Batch evaluate non-terminal leaves
            let values = self.batch_evaluate_leaves(&leaves);

            // Phase 3: Expand and backpropagate
            for (i, (path, players, leaf_game)) in leaves.into_iter().enumerate() {
                let value = values[i];
                self.expand_and_backup_typed(&mut root, &path, &players, &leaf_game, value);
            }

            simulation = batch_end;

            if self.should_stop_early(&root, simulation, start) {
                break;
            }
        }

        self.get_move_probabilities(&root)
    }

    /// Select a leaf node by traversing the tree (read-only).
    ///
    /// Returns the path (child indices), the player to move at each node along the way
    /// (index 0 is the root) and the game state at the leaf. パスがあるため手番は
    /// 経路長のパリティでは求まらないので、実際の局面から記録する。
    fn select_leaf_typed<G>(&self, root: &MctsNode, game: &G) -> (Vec<usize>, Vec<Player>, G)
    where
        G: Game + Clone,
    {
        let mut path: Vec<usize> = Vec::with_capacity(64);
        let mut players: Vec<Player> = Vec::with_capacity(65);
        players.push(game.current_player());

        let mut game_copy = game.clone();
        let mut current = root;

        while !current.children.is_empty() {
            let child_index = self.select_best_child_index(current);
            let move_idx = current.children[child_index].move_idx as usize;
            path.push(child_index);
            game_copy.make_move(move_idx / BOARD_SIZE, move_idx % BOARD_SIZE);
            current = &current.children[child_index].node;
            players.push(game_copy.current_player());
        }

        (path, players, game_copy)
    }

    /// Batch evaluate multiple leaf positions.
    /// Returns a value for each leaf (terminal leaves get exact values, others get NN eval).
    fn batch_evaluate_leaves<G>(&self, leaves: &[(Vec<usize>, Vec<Player>, G)]) -> Vec<f32>
    where
        G: Game + Clone,
    {
        let mut values = vec![0.0f32; leaves.len()];
        let mut nn_indices: Vec<usize> = Vec::new();
        let mut nn_games: Vec<&G> = Vec::new();

        // Separate terminal leaves from those needing NN evaluation
        for (i, (_, _, leaf_game)) in leaves.iter().enumerate() {
            if leaf_game.is_game_over() {
                values[i] = self.get_terminal_value(leaf_game, leaf_game.current_player());
            } else {
                // Check cache first
                let key = Self::board_key(leaf_game);
                let cache = self.eval_cache.borrow();
                if let Some(cached) = cache.get(&key) {
                    values[i] = cached.value;
                    drop(cache);
                } else {
                    drop(cache);
                    nn_indices.push(i);
                    nn_games.push(leaf_game);
                }
            }
        }

        // Batch evaluate all uncached positions in one forward pass
        if !nn_games.is_empty() {
            let batch_results = self.batch_evaluate_games(&nn_games);

            for (batch_idx, &leaf_idx) in nn_indices.iter().enumerate() {
                let (value, policy) = &batch_results[batch_idx];
                values[leaf_idx] = *value;

                // Cache the result
                let (_, _, leaf_game) = &leaves[leaf_idx];
                let key = Self::board_key(leaf_game);
                let mut cache = self.eval_cache.borrow_mut();
                if cache.len() >= self.config.eval_cache_size {
                    cache.clear();
                }
                cache.insert(
                    key,
                    CachedEvaluation {
                        value: *value,
                        policy_probs: policy.clone(),
                    },
                );
            }
        }

        values
    }

    /// Evaluate multiple game positions in a single batched forward pass.
    fn batch_evaluate_games<G: Game + ?Sized>(&self, games: &[&G]) -> Vec<(f32, Vec<f32>)> {
        let batch_size = games.len();

        // Build flat tensor data for all games: [batch, 3, 8, 8]
        let channels = 3 * BOARD_SIZE * BOARD_SIZE;
        let mut flat = Vec::with_capacity(batch_size * channels);

        for game in games {
            let board = game.board();
            let current = game.current_player();

            for row in board.iter() {
                for &cell in row.iter() {
                    // Channel 0: own stones
                    let own = match cell {
                        Cell::Black if current == Player::Black => 1.0f32,
                        Cell::White if current == Player::White => 1.0f32,
                        _ => 0.0f32,
                    };
                    flat.push(own);
                }
            }
            for row in board.iter() {
                for &cell in row.iter() {
                    // Channel 1: opponent stones
                    let opp = match cell {
                        Cell::Black if current == Player::White => 1.0f32,
                        Cell::White if current == Player::Black => 1.0f32,
                        _ => 0.0f32,
                    };
                    flat.push(opp);
                }
            }
            for row in board.iter() {
                for &cell in row.iter() {
                    // Channel 2: empty
                    let empty = if cell == Cell::Empty { 1.0f32 } else { 0.0f32 };
                    flat.push(empty);
                }
            }
        }

        let input = Tensor::from_data(
            TensorData::new(flat, [batch_size, 3, BOARD_SIZE, BOARD_SIZE]),
            &self.device,
        );

        let (policy_logits, value_tensor) = self.model.forward(input);
        let policy_probs = softmax(policy_logits, 1);

        let value_data = value_tensor.to_data();
        let policy_data = policy_probs.to_data();

        let value_slice = value_data.as_slice::<f32>().unwrap();
        let policy_slice = policy_data.as_slice::<f32>().unwrap();

        let num_moves = BOARD_SIZE * BOARD_SIZE;
        let mut results = Vec::with_capacity(batch_size);

        for (i, &value) in value_slice.iter().enumerate().take(batch_size) {
            let policy_start = i * num_moves;
            let policy: Vec<f32> = policy_slice[policy_start..policy_start + num_moves].to_vec();
            results.push((value, policy));
        }

        results
    }

    /// Expand a leaf node and backpropagate the value.
    fn expand_and_backup_typed<G>(
        &self,
        root: &mut MctsNode,
        path: &[usize],
        players: &[Player],
        leaf_game: &G,
        leaf_value: f32,
    ) where
        G: Game + Clone,
    {
        // 経路ノードの手番を確定させ、必要ならリーフを展開する
        {
            let mut current = &mut *root;
            if let Some(&player) = players.first() {
                current.player = player;
            }

            for (index, &child_index) in path.iter().enumerate() {
                current = &mut current.children[child_index].node;
                if let Some(&player) = players.get(index + 1) {
                    current.player = player;
                }
            }

            // Expand the leaf (if not terminal and not yet expanded)
            if !leaf_game.is_game_over() && current.children.is_empty() {
                let key = Self::board_key(leaf_game);
                let cache = self.eval_cache.borrow();
                if let Some(cached) = cache.get(&key) {
                    let valid_moves = self.get_valid_move_indices(leaf_game);
                    current.children = valid_moves
                        .iter()
                        .map(|&move_idx| MctsChild {
                            move_idx,
                            node: MctsNode::new(cached.policy_probs[move_idx as usize]),
                        })
                        .collect();
                }
                drop(cache);
            }
        }

        // Backpropagate
        Self::backup(root, path, leaf_game.current_player(), leaf_value);
    }

    /// Select best child using PUCT from parent perspective.
    fn select_best_child_index(&self, node: &MctsNode) -> usize {
        let mut best_index = 0usize;
        let mut best_score = f32::MIN;
        let parent_sqrt = (node.visit_count.max(1) as f32).sqrt();

        for (index, child) in node.children.iter().enumerate() {
            // child.q_value は子ノードの手番視点の値なので、手番が同じ（パス）なら
            // そのまま、違うなら符号を反転して親の視点に合わせる。
            let q = if child.node.player == node.player {
                child.node.q_value()
            } else {
                -child.node.q_value()
            };
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
            let mut node = MctsNode::new_terminal(value);
            node.player = game.current_player();
            return node;
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
        node.player = game.current_player();
        node.total_value = value;
        node.children = children;
        node
    }

    /// Backup values up the tree.
    ///
    /// 値の符号は経路長のパリティではなく、各ノードの手番（[`MctsNode::player`]）と
    /// リーフの手番が一致するかで決める。オセロはパスがあり 1 エッジで手番が
    /// 入れ替わらないことがあるため、パリティでは誤った符号になる。
    fn backup(root: &mut MctsNode, path: &[usize], leaf_player: Player, leaf_value: f32) {
        root.visit_count += 1;
        root.total_value += orient_for(root.player, leaf_player, leaf_value);

        let mut current = root;
        for &child_index in path {
            current = &mut current.children[child_index].node;
            current.visit_count += 1;
            current.total_value += orient_for(current.player, leaf_player, leaf_value);
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
    fn logits_to_probs(&self, logits: &Tensor<2>) -> Vec<f32> {
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
    pub fn device(&self) -> &Device {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn child(move_idx: u8, player: Player) -> MctsChild {
        let mut node = MctsNode::new(0.5);
        node.player = player;
        MctsChild { move_idx, node }
    }

    /// パスで手番が入れ替わらない枝では、値の符号が反転しないこと
    #[test]
    fn backup_orientation_handles_forced_passes() {
        // 根（黒）→ 子（パスにより手番が戻ってきた黒）
        let mut root = MctsNode::new(0.5);
        root.player = Player::Black;
        root.children
            .push(child((2 * BOARD_SIZE + 3) as u8, Player::Black));

        MctsSearch::backup(&mut root, &[0], Player::Black, 1.0);

        assert_eq!(root.visit_count, 1);
        assert_eq!(root.children[0].node.visit_count, 1);
        assert_eq!(root.total_value, 1.0);
        assert_eq!(root.children[0].node.total_value, 1.0);
    }

    /// 手番が入れ替わる通常の枝では符号が反転すること
    #[test]
    fn backup_orientation_alternates_players() {
        let mut root = MctsNode::new(0.5);
        root.player = Player::Black;
        root.children
            .push(child((2 * BOARD_SIZE + 3) as u8, Player::White));

        // 白のリーフが +1（白視点）→ 黒の根では -1
        MctsSearch::backup(&mut root, &[0], Player::White, 1.0);

        assert_eq!(root.total_value, -1.0);
        assert_eq!(root.children[0].node.total_value, 1.0);
    }

    /// 選択時の Q 値も手番の一致で符号が決まること
    #[test]
    fn orient_for_uses_player_identity() {
        assert_eq!(orient_for(Player::Black, Player::Black, 0.5), 0.5);
        assert_eq!(orient_for(Player::White, Player::Black, 0.5), -0.5);
    }
}
