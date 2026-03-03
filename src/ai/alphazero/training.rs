//! Training module for AlphaZero
//!
//! Implements policy/value joint optimization with Adam.

use crate::ai::alphazero::model::AlphaZeroModel;
use crate::ai::alphazero::self_play::TrainingExample;
use crate::game::BOARD_SIZE;

use burn::module::{AutodiffModule, Module};
use burn::nn::loss::{MseLoss, Reduction};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::activation::log_softmax;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::{Tensor, TensorData};
use rand::seq::SliceRandom;

/// Configuration for training
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    /// Number of epochs
    pub num_epochs: usize,
    /// Batch size
    pub batch_size: usize,
    /// Initial learning rate
    pub learning_rate: f64,
    /// Learning rate decay factor
    pub lr_decay: f64,
    /// Number of epochs before decaying LR
    pub lr_decay_epochs: usize,
    /// Max number of training batches per epoch
    pub steps_per_epoch: usize,
    /// Validation split
    pub validation_split: f32,
    /// Checkpoint directory
    pub checkpoint_dir: Option<String>,
    /// Save checkpoint every N epochs
    pub save_every: usize,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            num_epochs: 80,
            batch_size: 256,
            learning_rate: 0.001,
            lr_decay: 0.5,
            lr_decay_epochs: 20,
            steps_per_epoch: 1_000,
            validation_split: 0.1,
            checkpoint_dir: Some("checkpoints".to_string()),
            save_every: 10,
        }
    }
}

/// Trainer with full GPU support
pub struct Trainer<B: AutodiffBackend> {
    config: TrainingConfig,
    model: Option<AlphaZeroModel<B>>,
    device: B::Device,
    mse_loss: MseLoss,
}

impl<B: AutodiffBackend> Trainer<B> {
    pub fn new(config: TrainingConfig, model: AlphaZeroModel<B>, device: B::Device) -> Self {
        Self {
            config,
            model: Some(model),
            device,
            mse_loss: MseLoss::new(),
        }
    }

    /// Train the model
    pub fn train(&mut self, training_data: &[TrainingExample]) {
        if training_data.is_empty() {
            println!("No training data available!");
            return;
        }

        println!("Starting training with {} examples", training_data.len());
        println!("Configuration:");
        println!("  - Epochs: {}", self.config.num_epochs);
        println!("  - Batch size: {}", self.config.batch_size);
        println!("  - Initial learning rate: {}", self.config.learning_rate);
        println!("  - Device: {:?}", std::any::type_name::<B>());

        let split_idx =
            (training_data.len() as f32 * (1.0 - self.config.validation_split)) as usize;
        let split_idx = split_idx.clamp(1, training_data.len());
        let train_data = &training_data[..split_idx];
        let val_data = &training_data[split_idx..];

        println!(
            "Training set: {}, Validation set: {}",
            train_data.len(),
            val_data.len()
        );

        let mut optimizer = AdamConfig::new().init::<B, AlphaZeroModel<B>>();
        let mut best_val_loss = f32::INFINITY;
        let mut previous_lr = self.learning_rate_at_epoch(0);

        for epoch in 0..self.config.num_epochs {
            let current_lr = self.learning_rate_at_epoch(epoch);
            if epoch == 0 || (current_lr - previous_lr).abs() > f64::EPSILON {
                println!("Epoch {}: learning rate = {:.6}", epoch + 1, current_lr);
                previous_lr = current_lr;
            }

            let mut shuffled_indices: Vec<usize> = (0..train_data.len()).collect();
            shuffled_indices.shuffle(&mut rand::rng());

            let train_loss =
                self.train_epoch(train_data, &shuffled_indices, &mut optimizer, current_lr);
            let val_loss = if !val_data.is_empty() {
                self.validate(val_data)
            } else {
                train_loss
            };

            if epoch % 5 == 0 || epoch + 1 == self.config.num_epochs {
                println!(
                    "Epoch {}/{} - Train Loss: {:.6}, Val Loss: {:.6}",
                    epoch + 1,
                    self.config.num_epochs,
                    train_loss,
                    val_loss
                );
            }

            if val_loss < best_val_loss {
                best_val_loss = val_loss;
                if let Some(ref checkpoint_dir) = self.config.checkpoint_dir {
                    println!("New best validation loss: {:.6}", best_val_loss);
                    self.save_model_stub(checkpoint_dir, "best_model");
                }
            }

            if (epoch + 1) % self.config.save_every == 0
                && let Some(ref checkpoint_dir) = self.config.checkpoint_dir
            {
                println!("Saving checkpoint at epoch {}", epoch + 1);
                self.save_model_stub(checkpoint_dir, &format!("checkpoint_epoch_{}", epoch + 1));
            }
        }

        println!("\nTraining complete!");
        println!("Best validation loss: {:.6}", best_val_loss);
    }

    fn train_epoch<O>(
        &mut self,
        data: &[TrainingExample],
        shuffled_indices: &[usize],
        optimizer: &mut O,
        learning_rate: f64,
    ) -> f32
    where
        O: Optimizer<AlphaZeroModel<B>, B>,
    {
        let mut total_loss = 0.0f32;

        let num_batches_total = data.len().div_ceil(self.config.batch_size);
        let num_batches = num_batches_total.min(self.config.steps_per_epoch.max(1));

        for batch_idx in 0..num_batches {
            let start = batch_idx * self.config.batch_size;
            let end = (start + self.config.batch_size).min(shuffled_indices.len());
            if start >= end {
                break;
            }

            let batch_indices = &shuffled_indices[start..end];
            let loss = self.train_step(data, batch_indices, optimizer, learning_rate);
            total_loss += loss;
        }

        if num_batches > 0 {
            total_loss / num_batches as f32
        } else {
            0.0
        }
    }

    fn train_step<O>(
        &mut self,
        data: &[TrainingExample],
        batch_indices: &[usize],
        optimizer: &mut O,
        learning_rate: f64,
    ) -> f32
    where
        O: Optimizer<AlphaZeroModel<B>, B>,
    {
        let (board, target_policy, target_value) = self.make_batch(data, batch_indices);

        let model_ref = self
            .model
            .as_ref()
            .expect("Model should always be available while training");
        let (policy_logits, value_pred) = model_ref.forward(board);

        let log_probs = log_softmax(policy_logits, 1);
        let policy_loss = (target_policy * log_probs).sum_dim(1).mean().neg();
        let value_loss = self
            .mse_loss
            .forward(value_pred, target_value, Reduction::Mean);
        let loss = policy_loss + value_loss;

        let loss_value = Self::tensor_scalar(loss.clone());
        let grads = GradientsParams::from_grads(loss.backward(), model_ref);

        let model = self
            .model
            .take()
            .expect("Model should always be available while training");
        self.model = Some(optimizer.step(learning_rate, model, grads));

        loss_value
    }

    /// Validate on validation set
    fn validate(&self, data: &[TrainingExample]) -> f32 {
        let mut total_loss = 0.0f32;
        let batch_size = self.config.batch_size;
        let num_batches = data.len().div_ceil(batch_size).min(20);

        for batch_idx in 0..num_batches {
            let start = batch_idx * batch_size;
            let end = (start + batch_size).min(data.len());
            if start >= end {
                break;
            }

            let batch_indices: Vec<usize> = (start..end).collect();
            let (board, target_policy, target_value) = self.make_batch(data, &batch_indices);

            let model = self
                .model
                .as_ref()
                .expect("Model should always be available while validating");
            let (policy_logits, value_pred) = model.forward(board);

            let log_probs = log_softmax(policy_logits, 1);
            let policy_loss = (target_policy * log_probs).sum_dim(1).mean().neg();
            let value_loss = self
                .mse_loss
                .forward(value_pred, target_value, Reduction::Mean);
            let loss = policy_loss + value_loss;

            total_loss += Self::tensor_scalar(loss);
        }

        if num_batches > 0 {
            total_loss / num_batches as f32
        } else {
            0.0
        }
    }

    fn make_batch(
        &self,
        data: &[TrainingExample],
        batch_indices: &[usize],
    ) -> (Tensor<B, 4>, Tensor<B, 2>, Tensor<B, 2>) {
        let batch_size = batch_indices.len();
        let board_len = 3 * BOARD_SIZE * BOARD_SIZE;
        let policy_len = BOARD_SIZE * BOARD_SIZE;

        let mut boards = Vec::with_capacity(batch_size * board_len);
        let mut policies = Vec::with_capacity(batch_size * policy_len);
        let mut values = Vec::with_capacity(batch_size);

        for &idx in batch_indices {
            let example = &data[idx];
            debug_assert_eq!(example.board_tensor.len(), board_len);
            debug_assert_eq!(example.target_policy.len(), policy_len);
            boards.extend_from_slice(&example.board_tensor);
            policies.extend_from_slice(&example.target_policy);
            values.push(example.value);
        }

        let board = Tensor::<B, 4>::from_data(
            TensorData::new(boards, [batch_size, 3, BOARD_SIZE, BOARD_SIZE]),
            &self.device,
        );
        let target_policy = Tensor::<B, 2>::from_data(
            TensorData::new(policies, [batch_size, policy_len]),
            &self.device,
        );
        let target_value =
            Tensor::<B, 2>::from_data(TensorData::new(values, [batch_size, 1]), &self.device);

        (board, target_policy, target_value)
    }

    fn tensor_scalar(tensor: Tensor<B, 1>) -> f32 {
        let data = tensor.to_data();
        let values = data.as_slice::<f32>().unwrap();
        values.first().copied().unwrap_or(0.0)
    }

    fn learning_rate_at_epoch(&self, epoch: usize) -> f64 {
        if self.config.lr_decay_epochs == 0 {
            return self.config.learning_rate;
        }
        let decay_steps = epoch / self.config.lr_decay_epochs;
        self.config.learning_rate * self.config.lr_decay.powi(decay_steps as i32)
    }

    /// Get the trained model
    pub fn model(&self) -> &AlphaZeroModel<B> {
        self.model
            .as_ref()
            .expect("Model should always be available when queried")
    }

    /// Take ownership of the model
    pub fn into_model(self) -> AlphaZeroModel<B> {
        self.model
            .expect("Model should always be available when consumed")
    }

    /// Save model to file using safetensors format
    fn save_model_stub(&self, checkpoint_dir: &str, filename: &str) {
        use burn::record::DefaultFileRecorder;
        use std::fs;

        if let Err(e) = fs::create_dir_all(checkpoint_dir) {
            eprintln!("Failed to create checkpoint directory: {}", e);
            return;
        }

        let path = format!("{}/{}", checkpoint_dir, filename);

        use burn::record::FullPrecisionSettings;
        let recorder = DefaultFileRecorder::<FullPrecisionSettings>::new();

        let model = self
            .model
            .as_ref()
            .expect("Model should always be available when saving")
            .valid();

        match model.save_file(&path, &recorder) {
            Ok(_) => println!("Model saved to {}", path),
            Err(e) => eprintln!("Failed to save model: {:?}", e),
        }
    }
}
