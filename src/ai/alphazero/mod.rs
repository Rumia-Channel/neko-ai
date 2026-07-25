//! AlphaZero implementation for Othello
//!
//! This module implements the AlphaZero architecture using Burn ML framework.
//! Supports standard training and artistic (翻弄) training with special rewards.

#![allow(dead_code)]
#![allow(unused_imports)]

pub mod artistic;
pub mod mcts;
pub mod model;
pub mod player;
pub mod self_play;
pub mod tensor_utils;
pub mod training;

pub use artistic::{ArtisticPattern, evaluate_artistic_reward};
pub use mcts::{MctsConfig, MctsSearch};
pub use model::{AlphaZeroModel, AlphaZeroModelConfig};
pub use self_play::{RewardMode, SelfPlayConfig, SelfPlayEngine, TrainingExample};
pub use training::{Trainer, TrainingConfig};
