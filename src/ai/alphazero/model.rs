use burn::config::Config;
use burn::module::Module;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::{Linear, LinearConfig, PaddingConfig2d, Relu};
use burn::store::ModuleRecord;
use burn::tensor::{Device, Tensor};
use std::path::Path;

use crate::game::BOARD_SIZE;

/// Configuration for AlphaZero neural network
#[derive(Config, Debug)]
pub struct AlphaZeroModelConfig {
    /// Number of residual blocks (default: 10)
    #[config(default = 10)]
    pub num_res_blocks: usize,

    /// Number of filters in convolutional layers (default: 256)
    #[config(default = 256)]
    pub num_filters: usize,
}

/// Residual block for AlphaZero architecture
#[derive(Module, Debug)]
pub struct ResidualBlock {
    conv1: Conv2d,
    conv2: Conv2d,
    relu: Relu,
}

impl ResidualBlock {
    pub fn new(config: &AlphaZeroModelConfig, device: &Device) -> Self {
        let num_filters = config.num_filters;

        let conv1 = Conv2dConfig::new([num_filters, num_filters], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .init(device);

        let conv2 = Conv2dConfig::new([num_filters, num_filters], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .init(device);

        Self {
            conv1,
            conv2,
            relu: Relu::new(),
        }
    }

    pub fn forward(&self, input: Tensor<4>) -> Tensor<4> {
        let residual = input.clone();

        let x = self.conv1.forward(input);
        let x = self.relu.forward(x);
        let x = self.conv2.forward(x);

        // Skip connection
        let x = x + residual;
        self.relu.forward(x)
    }
}

/// AlphaZero neural network with policy and value heads
#[derive(Module, Debug)]
pub struct AlphaZeroModel {
    /// Initial convolution layer
    initial_conv: Conv2d,
    relu: Relu,

    /// Residual blocks
    res_blocks: Vec<ResidualBlock>,

    /// Policy head: outputs move probabilities
    policy_conv: Conv2d,
    policy_fc: Linear,

    /// Value head: outputs board evaluation
    value_conv: Conv2d,
    value_fc1: Linear,
    value_fc2: Linear,

    /// Number of possible moves (BOARD_SIZE * BOARD_SIZE)
    num_moves: usize,
}

impl AlphaZeroModel {
    pub fn new(config: &AlphaZeroModelConfig, device: &Device) -> Self {
        let num_filters = config.num_filters;
        let num_res_blocks = config.num_res_blocks;
        let board_size = BOARD_SIZE;
        let num_moves = board_size * board_size;

        // Initial convolution: 3 channels (own, opponent, empty) -> num_filters
        let initial_conv = Conv2dConfig::new([3, num_filters], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .init(device);

        // Residual blocks
        let mut res_blocks = Vec::with_capacity(num_res_blocks);
        for _ in 0..num_res_blocks {
            res_blocks.push(ResidualBlock::new(config, device));
        }

        // Policy head
        let policy_conv = Conv2dConfig::new([num_filters, 2], [1, 1]).init(device);
        let policy_fc = LinearConfig::new(2 * board_size * board_size, num_moves).init(device);

        // Value head
        let value_conv = Conv2dConfig::new([num_filters, 1], [1, 1]).init(device);
        let value_fc1 = LinearConfig::new(board_size * board_size, 256).init(device);
        let value_fc2 = LinearConfig::new(256, 1).init(device);

        Self {
            initial_conv,
            relu: Relu::new(),
            res_blocks,
            policy_conv,
            policy_fc,
            value_conv,
            value_fc1,
            value_fc2,
            num_moves,
        }
    }

    /// Forward pass returning both policy and value
    pub fn forward(&self, input: Tensor<4>) -> (Tensor<2>, Tensor<2>) {
        // Input: [batch, 3, 8, 8]
        let batch_size = input.dims()[0];

        // Initial convolution
        let mut x = self.initial_conv.forward(input);
        x = self.relu.forward(x);

        // Residual blocks
        for block in &self.res_blocks {
            x = block.forward(x);
        }

        // Policy head
        let policy = self.policy_conv.forward(x.clone());
        let policy = self.relu.forward(policy);
        let policy = policy.reshape([batch_size, 2 * BOARD_SIZE * BOARD_SIZE]);
        let policy_logits = self.policy_fc.forward(policy);

        // Value head
        let value = self.value_conv.forward(x);
        let value = self.relu.forward(value);
        let value = value.reshape([batch_size, BOARD_SIZE * BOARD_SIZE]);
        let value = self.value_fc1.forward(value);
        let value = self.relu.forward(value);
        let value = self.value_fc2.forward(value);
        let value = value.tanh(); // Output in range [-1, 1]

        (policy_logits, value)
    }

    /// Get policy (move probabilities) only
    pub fn forward_policy(&self, input: Tensor<4>) -> Tensor<2> {
        let batch_size = input.dims()[0];

        let mut x = self.initial_conv.forward(input);
        x = self.relu.forward(x);

        for block in &self.res_blocks {
            x = block.forward(x);
        }

        let policy = self.policy_conv.forward(x);
        let policy = self.relu.forward(policy);
        let policy = policy.reshape([batch_size, 2 * BOARD_SIZE * BOARD_SIZE]);
        self.policy_fc.forward(policy)
    }

    /// Get value (board evaluation) only
    pub fn forward_value(&self, input: Tensor<4>) -> Tensor<2> {
        let batch_size = input.dims()[0];

        let mut x = self.initial_conv.forward(input);
        x = self.relu.forward(x);

        for block in &self.res_blocks {
            x = block.forward(x);
        }

        let value = self.value_conv.forward(x);
        let value = self.relu.forward(value);
        let value = value.reshape([batch_size, BOARD_SIZE * BOARD_SIZE]);
        let value = self.value_fc1.forward(value);
        let value = self.relu.forward(value);
        let value = self.value_fc2.forward(value);
        value.tanh()
    }

    pub fn num_moves(&self) -> usize {
        self.num_moves
    }

    /// Load a trained model from a burnpack file (`.bpk`).
    ///
    /// Returns None if the file does not exist or loading fails.
    /// NOTE: burn 0.22 では記録形式が burnpack に変わり、旧 `.mpk` (MessagePack) の
    /// チェックポイントは読み込めない（再学習が必要）。
    pub fn load_trained(
        config: &AlphaZeroModelConfig,
        path: &str,
        device: &Device,
    ) -> Option<Self> {
        if !Path::new(path).exists() {
            return None;
        }

        let record = match ModuleRecord::load(path) {
            Ok(record) => record,
            Err(e) => {
                eprintln!("Failed to load model from {}: {:?}", path, e);
                return None;
            }
        };

        let model = Self::new(config, device);
        Some(model.load_record(record))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::alphazero::mcts::{MctsConfig, MctsSearch};
    use crate::ai::alphazero::tensor_utils::game_to_tensor;
    use crate::game::OthelloGame;

    /// Flex デバイスで順伝播と MCTS が動作すること（burn 0.22 移行のスモークテスト）
    #[test]
    fn forward_and_mcts_search_run_on_flex_device() {
        let device = Device::flex();
        let config = AlphaZeroModelConfig::new()
            .with_num_res_blocks(1)
            .with_num_filters(8);
        let model = AlphaZeroModel::new(&config, &device);

        let game = OthelloGame::default();
        let input = game_to_tensor(&game, &device);
        let (policy_logits, value) = model.forward(input);
        assert_eq!(policy_logits.dims(), [1, BOARD_SIZE * BOARD_SIZE]);
        assert_eq!(value.dims(), [1, 1]);

        // 価値は tanh 出力なので [-1, 1] に収まる
        let value_data = value.to_data();
        let value = value_data.as_slice::<f32>().unwrap()[0];
        assert!((-1.0..=1.0).contains(&value), "value out of range: {value}");

        let mcts_config = MctsConfig {
            num_simulations: 12,
            add_root_dirichlet_noise: false,
            ..Default::default()
        };
        let mcts = MctsSearch::new(mcts_config, model, device);
        let moves = mcts.search(&game);

        // 初期局面の有効手は 4 つ
        assert_eq!(moves.len(), 4, "unexpected moves: {moves:?}");
        let total: f32 = moves.iter().map(|(_, _, p)| *p).sum();
        assert!(
            (total - 1.0).abs() < 1e-3,
            "probabilities should sum to 1, got {total}"
        );
    }
}
