use burn::config::Config;
use burn::module::Module;
use burn::nn::{Linear, LinearConfig, Relu, conv::Conv2d, conv::Conv2dConfig};
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;
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
pub struct ResidualBlock<B: Backend> {
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
    relu: Relu,
}

impl<B: Backend> ResidualBlock<B> {
    pub fn new(config: &AlphaZeroModelConfig, device: &B::Device) -> Self {
        let num_filters = config.num_filters;

        let conv1 = Conv2dConfig::new([num_filters, num_filters], [3, 3])
            .with_padding(burn::nn::PaddingConfig2d::Explicit(1, 1))
            .init(device);

        let conv2 = Conv2dConfig::new([num_filters, num_filters], [3, 3])
            .with_padding(burn::nn::PaddingConfig2d::Explicit(1, 1))
            .init(device);

        Self {
            conv1,
            conv2,
            relu: Relu::new(),
        }
    }

    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
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
pub struct AlphaZeroModel<B: Backend> {
    /// Initial convolution layer
    initial_conv: Conv2d<B>,
    relu: Relu,

    /// Residual blocks
    res_blocks: Vec<ResidualBlock<B>>,

    /// Policy head: outputs move probabilities
    policy_conv: Conv2d<B>,
    policy_fc: Linear<B>,

    /// Value head: outputs board evaluation
    value_conv: Conv2d<B>,
    value_fc1: Linear<B>,
    value_fc2: Linear<B>,

    /// Number of possible moves (BOARD_SIZE * BOARD_SIZE)
    num_moves: usize,
}

impl<B: Backend> AlphaZeroModel<B> {
    pub fn new(config: &AlphaZeroModelConfig, device: &B::Device) -> Self {
        let num_filters = config.num_filters;
        let num_res_blocks = config.num_res_blocks;
        let board_size = BOARD_SIZE;
        let num_moves = board_size * board_size;

        // Initial convolution: 3 channels (own, opponent, empty) -> num_filters
        let initial_conv = Conv2dConfig::new([3, num_filters], [3, 3])
            .with_padding(burn::nn::PaddingConfig2d::Explicit(1, 1))
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
    pub fn forward(&self, input: Tensor<B, 4>) -> (Tensor<B, 2>, Tensor<B, 2>) {
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
    pub fn forward_policy(&self, input: Tensor<B, 4>) -> Tensor<B, 2> {
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
    pub fn forward_value(&self, input: Tensor<B, 4>) -> Tensor<B, 2> {
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
}

/// Training batch item
#[derive(Debug, Clone)]
pub struct TrainingItem {
    pub board_tensor: Vec<f32>,
    pub target_policy: Vec<f32>,
    pub target_value: f32,
}

/// Training batch structure
#[derive(Debug, Clone)]
pub struct TrainingBatch<B: Backend> {
    pub board: Tensor<B, 4>,
    pub target_policy: Tensor<B, 2>,
    pub target_value: Tensor<B, 2>,
}
