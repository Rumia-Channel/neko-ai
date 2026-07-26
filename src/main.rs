#![recursion_limit = "256"]

mod ai;
mod app;
mod game;

use ai::alphazero::{
    AlphaZeroModel, AlphaZeroModelConfig, MctsConfig, MctsSearch, RewardMode, SelfPlayConfig,
    SelfPlayEngine, Trainer, TrainingConfig,
};
use app::GameApp;
use eframe::egui;
use std::env;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TrainingProfile {
    Fast,
    Balanced,
    HighQuality,
}

impl TrainingProfile {
    fn from_args(args: &[String]) -> Self {
        for arg in args {
            match arg.as_str() {
                "--fast" | "--training-fast" | "--profile=fast" => return Self::Fast,
                "--balanced" | "--training-balanced" | "--profile=balanced" => {
                    return Self::Balanced;
                }
                "--high" | "--training-high" | "--profile=high" | "--profile=high-quality" => {
                    return Self::HighQuality;
                }
                _ => {}
            }
        }

        // Default to high-quality training.
        Self::HighQuality
    }

    fn as_str(&self) -> &'static str {
        match self {
            Self::Fast => "Fast",
            Self::Balanced => "Balanced",
            Self::HighQuality => "HighQuality",
        }
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let profile = TrainingProfile::from_args(&args);

    let is_gpu = args.iter().any(|arg| arg.ends_with("-gpu"));
    let is_honrou = args
        .iter()
        .any(|arg| arg.contains("honrou"));
    let is_training = args.iter().any(|arg| arg.starts_with("--training"));

    if is_training && is_honrou && is_gpu {
        run_training_mode_honrou_gpu(profile);
    } else if is_training && is_honrou {
        run_training_mode_honrou(profile);
    } else if is_training && is_gpu {
        run_training_mode_gpu(profile);
    } else if is_training {
        run_training_mode(profile);
    } else if let Err(e) = run_gui_mode() {
        eprintln!("Error running GUI: {}", e);
        std::process::exit(1);
    }
}

fn run_gui_mode() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([600.0, 700.0]),
        ..Default::default()
    };

    eframe::run_native(
        "オセロ",
        options,
        Box::new(|cc| {
            // Load Japanese font
            let mut fonts = egui::FontDefinitions::default();

            // Load the custom font data
            let font_data = include_bytes!("../fonts/BIZ_UDPGothic/BIZUDPGothic-Regular.ttf");
            fonts.font_data.insert(
                "japanese_font".to_owned(),
                egui::FontData::from_owned(font_data.to_vec()).into(),
            );

            // Add the font to the proportional and monospace families
            fonts
                .families
                .entry(egui::FontFamily::Proportional)
                .or_default()
                .insert(0, "japanese_font".to_owned());
            fonts
                .families
                .entry(egui::FontFamily::Monospace)
                .or_default()
                .push("japanese_font".to_owned());

            cc.egui_ctx.set_fonts(fonts);

            Ok(Box::new(GameApp::default()))
        }),
    )
}

fn build_self_play_config(profile: TrainingProfile, is_gpu: bool) -> SelfPlayConfig {
    let (num_games, num_simulations, checkpoint_interval) = match (profile, is_gpu) {
        (TrainingProfile::Fast, false) => (24, 160, 4),
        (TrainingProfile::Balanced, false) => (80, 420, 8),
        (TrainingProfile::HighQuality, false) => (220, 900, 10),
        (TrainingProfile::Fast, true) => (64, 320, 8),
        (TrainingProfile::Balanced, true) => (240, 900, 12),
        (TrainingProfile::HighQuality, true) => (900, 1_600, 20),
    };

    let num_mixed_games = match (profile, is_gpu) {
        (TrainingProfile::Fast, false) => 8,
        (TrainingProfile::Balanced, false) => 20,
        (TrainingProfile::HighQuality, false) => 50,
        (TrainingProfile::Fast, true) => 16,
        (TrainingProfile::Balanced, true) => 40,
        (TrainingProfile::HighQuality, true) => 100,
    };

    SelfPlayConfig {
        num_games,
        num_mixed_games,
        temperature: 1.0,
        temp_threshold: 30,
        temp_after_threshold: 0.05,
        checkpoint_interval,
        mcts_config: MctsConfig {
            num_simulations,
            dirichlet_alpha: 0.3,
            dirichlet_weight: 0.25,
            temperature: 1.0,
            c_puct: 1.3,
            add_root_dirichlet_noise: true,
            enable_progress_log: false,
            eval_cache_size: 120_000,
            skip_single_move_search: true,
            max_search_time_ms: None,
            min_simulations_before_stop: num_simulations / 2,
            early_stop_visit_ratio: Some(0.97),
        },
        augment_symmetry: true,
        max_moves_per_game: 120,
        reward_mode: RewardMode::Standard,
    }
}

fn build_training_config(profile: TrainingProfile, is_gpu: bool) -> TrainingConfig {
    match (profile, is_gpu) {
        (TrainingProfile::Fast, false) => TrainingConfig {
            num_epochs: 20,
            batch_size: 128,
            learning_rate: 0.001,
            lr_decay: 0.7,
            lr_decay_epochs: 8,
            steps_per_epoch: 300,
            validation_split: 0.1,
            checkpoint_dir: Some("checkpoints".to_string()),
            save_every: 5,
            model_name: "best_model".to_string(),
        },
        (TrainingProfile::Balanced, false) => TrainingConfig {
            num_epochs: 70,
            batch_size: 192,
            learning_rate: 0.001,
            lr_decay: 0.6,
            lr_decay_epochs: 15,
            steps_per_epoch: 1_200,
            validation_split: 0.1,
            checkpoint_dir: Some("checkpoints".to_string()),
            save_every: 10,
            model_name: "best_model".to_string(),
        },
        (TrainingProfile::HighQuality, false) => TrainingConfig {
            num_epochs: 140,
            batch_size: 256,
            learning_rate: 0.0008,
            lr_decay: 0.5,
            lr_decay_epochs: 20,
            steps_per_epoch: 2_200,
            validation_split: 0.1,
            checkpoint_dir: Some("checkpoints".to_string()),
            save_every: 10,
            model_name: "best_model".to_string(),
        },
        (TrainingProfile::Fast, true) => TrainingConfig {
            num_epochs: 40,
            batch_size: 256,
            learning_rate: 0.001,
            lr_decay: 0.6,
            lr_decay_epochs: 12,
            steps_per_epoch: 800,
            validation_split: 0.1,
            checkpoint_dir: Some("checkpoints".to_string()),
            save_every: 5,
            model_name: "best_model".to_string(),
        },
        (TrainingProfile::Balanced, true) => TrainingConfig {
            num_epochs: 110,
            batch_size: 512,
            learning_rate: 0.001,
            lr_decay: 0.5,
            lr_decay_epochs: 20,
            steps_per_epoch: 2_000,
            validation_split: 0.1,
            checkpoint_dir: Some("checkpoints".to_string()),
            save_every: 10,
            model_name: "best_model".to_string(),
        },
        (TrainingProfile::HighQuality, true) => TrainingConfig {
            num_epochs: 220,
            batch_size: 768,
            learning_rate: 0.0007,
            lr_decay: 0.5,
            lr_decay_epochs: 25,
            steps_per_epoch: 3_000,
            validation_split: 0.1,
            checkpoint_dir: Some("checkpoints".to_string()),
            save_every: 10,
            model_name: "best_model".to_string(),
        },
    }
}

fn run_training_mode(profile: TrainingProfile) {
    println!("========================================");
    println!("  AlphaZero Training Mode (CPU)");
    println!("========================================\n");
    println!("Training profile: {}\n", profile.as_str());

    use burn::backend::ndarray::NdArrayDevice;

    type InferBackend = burn::backend::ndarray::NdArray;
    type TrainBackend = burn::backend::Autodiff<InferBackend>;

    // Setup device
    let device = NdArrayDevice::default();
    println!("Using device: CPU (NdArray)");

    // Create model configuration
    let model_config = AlphaZeroModelConfig::new();
    println!("Initializing AlphaZero model...");
    println!("  - Residual blocks: {}", model_config.num_res_blocks);
    println!("  - Filters: {}", model_config.num_filters);

    let self_play_model = AlphaZeroModel::<InferBackend>::new(&model_config, &device);

    let self_play_config = build_self_play_config(profile, false);
    println!("\nSelf-play configuration:");
    println!("  - Games: {}", self_play_config.num_games);
    println!(
        "  - MCTS simulations: {}",
        self_play_config.mcts_config.num_simulations
    );
    println!(
        "  - Symmetry augmentation: {}",
        self_play_config.augment_symmetry
    );

    println!("\nInitializing self-play engine...");
    let mcts = MctsSearch::new(self_play_config.mcts_config, self_play_model, device);
    let self_play = SelfPlayEngine::new(self_play_config, mcts, device);

    println!("\n--- Starting Self-Play ---");
    let training_data = self_play.generate_data();

    println!("\n--- Starting Training ---");
    let training_model = AlphaZeroModel::<TrainBackend>::new(&model_config, &device);
    let training_config = build_training_config(profile, false);

    let mut trainer = Trainer::new(training_config, training_model, device);
    trainer.train(&training_data);

    println!("\n========================================");
    println!("  Training Complete!");
    println!("========================================");
}

#[allow(unexpected_cfgs)]
fn run_training_mode_gpu(profile: TrainingProfile) {
    println!("========================================");
    println!("  AlphaZero Training Mode (GPU)");
    println!("========================================\n");
    println!("Training profile: {}\n", profile.as_str());

    #[cfg(feature = "wgpu")]
    {
        use burn::backend::wgpu::WgpuDevice;

        type InferBackend = burn::backend::wgpu::Wgpu;
        type TrainBackend = burn::backend::Autodiff<InferBackend>;

        // Setup GPU device
        let device = WgpuDevice::default();
        println!("Using device: GPU (WGPU)");

        let model_config = AlphaZeroModelConfig::new();
        println!("Initializing AlphaZero model...");
        println!("  - Residual blocks: {}", model_config.num_res_blocks);
        println!("  - Filters: {}", model_config.num_filters);

        let self_play_model = AlphaZeroModel::<InferBackend>::new(&model_config, &device);
        let self_play_config = build_self_play_config(profile, true);
        println!("\nSelf-play configuration:");
        println!("  - Games: {}", self_play_config.num_games);
        println!(
            "  - MCTS simulations: {}",
            self_play_config.mcts_config.num_simulations
        );
        println!(
            "  - Symmetry augmentation: {}",
            self_play_config.augment_symmetry
        );

        println!("\nInitializing self-play engine with GPU...");
        let mcts = MctsSearch::new(
            self_play_config.mcts_config,
            self_play_model,
            device.clone(),
        );
        let self_play = SelfPlayEngine::new(self_play_config, mcts, device.clone());

        println!("\n--- Starting Self-Play ---");
        let training_data = self_play.generate_data();

        println!("\n--- Starting Training on GPU ---");
        let training_model = AlphaZeroModel::<TrainBackend>::new(&model_config, &device);
        let training_config = build_training_config(profile, true);

        let mut trainer = Trainer::new(training_config, training_model, device);
        trainer.train(&training_data);

        println!("\n========================================");
        println!("  Training Complete!");
        println!("========================================");
    }

    #[cfg(not(feature = "wgpu"))]
    {
        println!("GPU support not enabled. Compile with --features wgpu");
        println!("Falling back to CPU mode...");
        run_training_mode(profile);
    }
}

fn run_training_mode_honrou(profile: TrainingProfile) {
    println!("========================================");
    println!("  AlphaZero 翻弄 Training Mode (CPU)");
    println!("  Artistic Reward: 芸術的報酬");
    println!("========================================\n");
    println!("Training profile: {}\n", profile.as_str());

    use burn::backend::ndarray::NdArrayDevice;

    type InferBackend = burn::backend::ndarray::NdArray;
    type TrainBackend = burn::backend::Autodiff<InferBackend>;

    let device = NdArrayDevice::default();
    println!("Using device: CPU (NdArray)");

    let model_config = AlphaZeroModelConfig::new();
    println!("Initializing AlphaZero model (artistic)...");
    println!("  - Residual blocks: {}", model_config.num_res_blocks);
    println!("  - Filters: {}", model_config.num_filters);

    let self_play_model = AlphaZeroModel::<InferBackend>::new(&model_config, &device);

    // Build artistic self-play config
    let mut self_play_config = build_self_play_config(profile, false);
    self_play_config.reward_mode = RewardMode::Artistic;
    // 翻弄はより多くの対局で芸術的パターンを学習
    self_play_config.num_games = match profile {
        TrainingProfile::Fast => 40,
        TrainingProfile::Balanced => 150,
        TrainingProfile::HighQuality => 400,
    };
    self_play_config.num_mixed_games = match profile {
        TrainingProfile::Fast => 16,
        TrainingProfile::Balanced => 40,
        TrainingProfile::HighQuality => 100,
    };

    println!("\nSelf-play configuration (artistic):");
    println!("  - Games: {}", self_play_config.num_games);
    println!(
        "  - Mixed opponent games: {} per difficulty",
        self_play_config.num_mixed_games
    );
    println!(
        "  - MCTS simulations: {}",
        self_play_config.mcts_config.num_simulations
    );
    println!("  - Reward mode: Artistic (芸術的報酬)");
    println!(
        "  - Symmetry augmentation: {}",
        self_play_config.augment_symmetry
    );

    println!("\nInitializing self-play engine (artistic)...");
    let mcts = MctsSearch::new(self_play_config.mcts_config, self_play_model, device);
    let self_play = SelfPlayEngine::new(self_play_config, mcts, device);

    println!("\n--- Starting Self-Play (Artistic) ---");
    let training_data = self_play.generate_data();

    println!("\n--- Starting Training (Artistic) ---");
    let training_model = AlphaZeroModel::<TrainBackend>::new(&model_config, &device);

    // 翻弄用トレーニング設定: より長い学習で芸術的パターンを定着
    let mut training_config = build_training_config(profile, false);
    training_config.checkpoint_dir = Some("checkpoints".to_string());
    training_config.model_name = "honrou_model".to_string();
    // 芸術的報酬は値の範囲が広いため、学習率を少し下げて安定化
    training_config.learning_rate *= 0.8;

    let mut trainer = Trainer::new(training_config, training_model, device);
    trainer.train(&training_data);

    // 翻弄モデルとして保存
    println!("\nSaving honrou model to checkpoints/honrou_model...");
    // Note: Trainer saves best_model internally; for honrou we rename conceptually.
    // The model is saved via trainer's checkpoint mechanism.

    println!("\n========================================");
    println!("  翻弄 Training Complete!");
    println!("  Artistic patterns learned.");
    println!("========================================");
}

#[allow(unexpected_cfgs)]
fn run_training_mode_honrou_gpu(profile: TrainingProfile) {
    println!("========================================");
    println!("  AlphaZero 翻弄 Training Mode (GPU)");
    println!("  Artistic Reward: 芸術的報酬");
    println!("========================================\n");
    println!("Training profile: {}\n", profile.as_str());

    #[cfg(feature = "wgpu")]
    {
        use burn::backend::wgpu::WgpuDevice;

        type InferBackend = burn::backend::wgpu::Wgpu;
        type TrainBackend = burn::backend::Autodiff<InferBackend>;

        let device = WgpuDevice::default();
        println!("Using device: GPU (WGPU)");

        let model_config = AlphaZeroModelConfig::new();
        println!("Initializing AlphaZero model (artistic)...");
        println!("  - Residual blocks: {}", model_config.num_res_blocks);
        println!("  - Filters: {}", model_config.num_filters);

        let self_play_model = AlphaZeroModel::<InferBackend>::new(&model_config, &device);

        let mut self_play_config = build_self_play_config(profile, true);
        self_play_config.reward_mode = RewardMode::Artistic;
        self_play_config.num_games = match profile {
            TrainingProfile::Fast => 80,
            TrainingProfile::Balanced => 300,
            TrainingProfile::HighQuality => 800,
        };
        self_play_config.num_mixed_games = match profile {
            TrainingProfile::Fast => 24,
            TrainingProfile::Balanced => 60,
            TrainingProfile::HighQuality => 150,
        };

        println!("\nSelf-play configuration (artistic):");
        println!("  - Games: {}", self_play_config.num_games);
        println!(
            "  - Mixed opponent games: {} per difficulty",
            self_play_config.num_mixed_games
        );
        println!(
            "  - MCTS simulations: {}",
            self_play_config.mcts_config.num_simulations
        );
        println!("  - Reward mode: Artistic (芸術的報酬)");
        println!(
            "  - Symmetry augmentation: {}",
            self_play_config.augment_symmetry
        );

        println!("\nInitializing self-play engine with GPU (artistic)...");
        let mcts = MctsSearch::new(
            self_play_config.mcts_config,
            self_play_model,
            device.clone(),
        );
        let self_play = SelfPlayEngine::new(self_play_config, mcts, device.clone());

        println!("\n--- Starting Self-Play (Artistic) ---");
        let training_data = self_play.generate_data();

        println!("\n--- Starting Training on GPU (Artistic) ---");
        let training_model = AlphaZeroModel::<TrainBackend>::new(&model_config, &device);

        let mut training_config = build_training_config(profile, true);
        training_config.checkpoint_dir = Some("checkpoints".to_string());
        training_config.model_name = "honrou_model".to_string();
        training_config.learning_rate *= 0.8;

        let mut trainer = Trainer::new(training_config, training_model, device);
        trainer.train(&training_data);

        println!("\n========================================");
        println!("  翻弄 Training Complete!");
        println!("  Artistic patterns learned.");
        println!("========================================");
    }

    #[cfg(not(feature = "wgpu"))]
    {
        println!("GPU support not enabled. Compile with --features wgpu");
        println!("Falling back to CPU mode...");
        run_training_mode_honrou(profile);
    }
}
