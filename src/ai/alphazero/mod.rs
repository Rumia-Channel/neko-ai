//! AlphaZero implementation for Othello
//!
//! This module implements the AlphaZero architecture using Burn ML framework.
//! Currently a simplified placeholder implementation.

#![allow(dead_code)]
#![allow(unused_imports)]

pub mod mcts;
pub mod model;
pub mod player;
pub mod self_play;
pub mod tensor_utils;
pub mod training;

pub use model::{AlphaZeroModel, AlphaZeroModelConfig};
pub use self_play::{SelfPlayConfig, SelfPlayEngine};
pub use training::{Trainer, TrainingConfig};
