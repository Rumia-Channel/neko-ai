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
    let is_honrou = args.iter().any(|arg| arg.contains("honrou"));
    let is_training = args.iter().any(|arg| arg.starts_with("--training"));
    let model_config = model_config_from_args(&args);

    if args.iter().any(|arg| arg == "--bench") {
        run_bench_mode(&args);
    } else if args.iter().any(|arg| arg == "--bench-gpu") {
        run_bench_gpu_mode(&args);
    } else if is_training && is_honrou && is_gpu {
        run_training_mode_honrou_gpu(profile, &model_config);
    } else if is_training && is_honrou {
        run_training_mode_honrou(profile, &model_config);
    } else if is_training && is_gpu {
        run_training_mode_gpu(profile, &model_config);
    } else if is_training {
        run_training_mode(profile, &model_config);
    } else if let Err(e) = run_gui_mode() {
        eprintln!("Error running GUI: {}", e);
        std::process::exit(1);
    }
}

/// 日本語表示に使うフォントの探索候補（先頭ほど優先）。
///
/// リポジトリ同梱の BIZ UDPGothic を最優先し、無ければ OS 標準の日本語フォントを使う。
fn japanese_font_candidates() -> Vec<std::path::PathBuf> {
    use std::path::PathBuf;

    let mut candidates = vec![PathBuf::from(
        "fonts/BIZ_UDPGothic/BIZUDPGothic-Regular.ttf",
    )];

    // 実行ファイル相対（配布バイナリや target/release からの起動用）
    if let Ok(exe) = std::env::current_exe()
        && let Some(dir) = exe.parent()
    {
        candidates.push(dir.join("fonts/BIZ_UDPGothic/BIZUDPGothic-Regular.ttf"));
        candidates.push(dir.join("../fonts/BIZ_UDPGothic/BIZUDPGothic-Regular.ttf"));
    }

    // OS 標準フォントのディレクトリ
    let font_dir = if cfg!(windows) {
        std::env::var("SystemRoot")
            .map(|root| PathBuf::from(root).join("Fonts"))
            .unwrap_or_else(|_| PathBuf::from("C:\\Windows\\Fonts"))
    } else if cfg!(target_os = "macos") {
        PathBuf::from("/System/Library/Fonts")
    } else {
        PathBuf::from("/usr/share/fonts")
    };

    for name in [
        "BIZ-UDGothicR.ttc", // Windows 同梱の BIZ UDGothic
        "meiryo.ttc",
        "YuGothM.ttc",
        "msgothic.ttc",
    ] {
        candidates.push(font_dir.join(name));
    }

    // Linux / macOS の代表的な日本語フォント
    for path in [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
        "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
    ] {
        candidates.push(PathBuf::from(path));
    }

    candidates
}

/// TrueType / OpenType のコンテナかどうかを先頭 4 バイトで判定する。
///
/// 壊れたファイルや HTML を掴んだまま egui に渡して panic するのを防ぐ。
fn looks_like_font(bytes: &[u8]) -> bool {
    matches!(
        bytes.get(..4),
        Some(b"\x00\x01\x00\x00" | b"OTTO" | b"true" | b"ttcf")
    )
}

/// 日本語フォントを探索して読み込む。見つかれば (パス, バイト列) を返す。
fn load_japanese_font() -> Option<(std::path::PathBuf, Vec<u8>)> {
    for path in japanese_font_candidates() {
        let Ok(bytes) = std::fs::read(&path) else {
            continue;
        };
        if looks_like_font(&bytes) {
            return Some((path, bytes));
        }
        eprintln!(
            "警告: '{}' はフォント形式ではないため読み飛ばします",
            path.display()
        );
    }
    None
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
            // 日本語フォントは実行時に探索して読み込む。
            // 同梱フォントは .gitignore 対象（*.ttf）のため include_bytes! で埋め込むと
            // フォント未配置の環境でコンパイル自体が失敗してしまう。
            let mut fonts = egui::FontDefinitions::default();

            if let Some((path, font_bytes)) = load_japanese_font() {
                println!("日本語フォントを読み込みました: {}", path.display());
                fonts.font_data.insert(
                    "japanese_font".to_owned(),
                    egui::FontData::from_owned(font_bytes).into(),
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
            } else {
                eprintln!(
                    "警告: 日本語フォントが見つかりません。日本語は正しく表示されません。\n\
                     フォントを配置する場合: fonts/BIZ_UDPGothic/BIZUDPGothic-Regular.ttf"
                );
            }

            cc.egui_ctx.set_fonts(fonts);

            Ok(Box::new(GameApp::default()))
        }),
    )
}

fn build_self_play_config(profile: TrainingProfile, is_gpu: bool) -> SelfPlayConfig {
    // CPU プロファイルはノート PC でも現実的な時間で終わる規模に抑える
    // （1 手あたりの推論コストに比例するため、sims 数が支配的）。
    let (num_games, num_simulations, checkpoint_interval) = match (profile, is_gpu) {
        (TrainingProfile::Fast, false) => (32, 120, 4),
        (TrainingProfile::Balanced, false) => (64, 200, 8),
        (TrainingProfile::HighQuality, false) => (128, 320, 10),
        (TrainingProfile::Fast, true) => (64, 320, 8),
        (TrainingProfile::Balanced, true) => (240, 900, 12),
        (TrainingProfile::HighQuality, true) => (900, 1_600, 20),
    };

    let num_mixed_games = match (profile, is_gpu) {
        (TrainingProfile::Fast, false) => 8,
        (TrainingProfile::Balanced, false) => 16,
        (TrainingProfile::HighQuality, false) => 32,
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
            eval_batch_size: 1,
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
            num_epochs: 40,
            batch_size: 256,
            learning_rate: 0.001,
            lr_decay: 0.6,
            lr_decay_epochs: 15,
            steps_per_epoch: 800,
            validation_split: 0.1,
            checkpoint_dir: Some("checkpoints".to_string()),
            save_every: 10,
            model_name: "best_model".to_string(),
        },
        (TrainingProfile::HighQuality, false) => TrainingConfig {
            num_epochs: 60,
            batch_size: 256,
            learning_rate: 0.0008,
            lr_decay: 0.5,
            lr_decay_epochs: 20,
            steps_per_epoch: 1_500,
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

/// 使用するモデル構成を引数から決める。
///
/// 既定は CPU（ノート PC でも回る）向けの軽量構成。大きい構成を使いたい場合は
/// `--model standard` を指定する。
fn model_config_from_args(args: &[String]) -> AlphaZeroModelConfig {
    for (index, arg) in args.iter().enumerate() {
        if arg != "--model" {
            continue;
        }
        match args.get(index + 1).map(|value| value.as_str()) {
            Some("standard") | Some("large") => return AlphaZeroModelConfig::standard(),
            Some("light") | Some("small") => return AlphaZeroModelConfig::light(),
            _ => {}
        }
    }

    AlphaZeroModelConfig::light()
}

/// 利用可能なスレッド数
fn available_threads() -> usize {
    std::thread::available_parallelism()
        .map(|threads| threads.get())
        .unwrap_or(1)
}

fn run_training_mode(profile: TrainingProfile, model_config: &AlphaZeroModelConfig) {
    println!("========================================");
    println!("  AlphaZero Training Mode (CPU)");
    println!("========================================\n");
    println!("Training profile: {}\n", profile.as_str());

    use burn::tensor::Device;

    // CPU 推論・学習には pure-Rust の Flex バックエンドを使う
    let device = Device::flex();
    println!("Using device: CPU (Flex), threads: {}", available_threads());

    // Create model configuration
    println!("Initializing AlphaZero model...");
    println!("  - Residual blocks: {}", model_config.num_res_blocks);
    println!("  - Filters: {}", model_config.num_filters);

    let self_play_model = AlphaZeroModel::new(model_config, &device);

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
    let mcts = MctsSearch::new(
        self_play_config.mcts_config,
        self_play_model,
        device.clone(),
    );
    let self_play = SelfPlayEngine::new(self_play_config, mcts, device.clone());

    println!("\n--- Starting Self-Play ---");
    let training_data = self_play.generate_data();

    println!("\n--- Starting Training ---");
    // 学習は autodiff を有効にしたデバイスで行う
    let training_device = device.clone().autodiff();
    let training_model = AlphaZeroModel::new(model_config, &training_device);
    let training_config = build_training_config(profile, false);

    let mut trainer = Trainer::new(
        training_config,
        training_model,
        model_config.clone(),
        training_device,
    );
    trainer.train(&training_data);

    println!("\n========================================");
    println!("  Training Complete!");
    println!("========================================");
}

#[allow(unexpected_cfgs)]
fn run_training_mode_gpu(profile: TrainingProfile, model_config: &AlphaZeroModelConfig) {
    println!("========================================");
    println!("  AlphaZero Training Mode (GPU)");
    println!("========================================\n");
    println!("Training profile: {}\n", profile.as_str());

    #[cfg(feature = "wgpu")]
    {
        use burn::tensor::{Device, DeviceKind};

        // Self-play uses CPU for inference — GPU batch_size=1 is slower
        // due to CPU↔GPU transfer overhead per simulation.
        let cpu_device = Device::flex();
        let gpu_device = Device::wgpu(DeviceKind::DefaultDevice);
        println!(
            "Self-play device: CPU (Flex), threads: {}",
            available_threads()
        );
        println!("Training device:  GPU (WGPU)");

        println!("Initializing AlphaZero model...");
        println!("  - Residual blocks: {}", model_config.num_res_blocks);
        println!("  - Filters: {}", model_config.num_filters);

        let self_play_model = AlphaZeroModel::new(model_config, &cpu_device);
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

        println!("\nInitializing self-play engine (CPU inference)...");
        let mcts = MctsSearch::new(
            self_play_config.mcts_config,
            self_play_model,
            cpu_device.clone(),
        );
        let self_play = SelfPlayEngine::new(self_play_config, mcts, cpu_device.clone());

        println!("\n--- Starting Self-Play ---");
        let training_data = self_play.generate_data();

        println!("\n--- Starting Training on GPU ---");
        let training_device = gpu_device.autodiff();
        let training_model = AlphaZeroModel::new(model_config, &training_device);
        let training_config = build_training_config(profile, true);

        let mut trainer = Trainer::new(
            training_config,
            training_model,
            model_config.clone(),
            training_device,
        );
        trainer.train(&training_data);

        println!("\n========================================");
        println!("  Training Complete!");
        println!("========================================");
    }

    #[cfg(not(feature = "wgpu"))]
    {
        println!("GPU support not enabled. Compile with --features wgpu");
        println!("Falling back to CPU mode...");
        run_training_mode(profile, model_config);
    }
}

fn run_training_mode_honrou(profile: TrainingProfile, model_config: &AlphaZeroModelConfig) {
    println!("========================================");
    println!("  AlphaZero 翻弄 Training Mode (CPU)");
    println!("  Artistic Reward: 芸術的報酬");
    println!("========================================\n");
    println!("Training profile: {}\n", profile.as_str());

    use burn::tensor::Device;

    let device = Device::flex();
    println!("Using device: CPU (Flex), threads: {}", available_threads());
    println!("Initializing AlphaZero model (artistic)...");
    println!("  - Residual blocks: {}", model_config.num_res_blocks);
    println!("  - Filters: {}", model_config.num_filters);

    let self_play_model = AlphaZeroModel::new(model_config, &device);

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
    let mcts = MctsSearch::new(
        self_play_config.mcts_config,
        self_play_model,
        device.clone(),
    );
    let self_play = SelfPlayEngine::new(self_play_config, mcts, device.clone());

    println!("\n--- Starting Self-Play (Artistic) ---");
    let training_data = self_play.generate_data();

    println!("\n--- Starting Training (Artistic) ---");
    let training_device = device.clone().autodiff();
    let training_model = AlphaZeroModel::new(model_config, &training_device);

    // 翻弄用トレーニング設定: より長い学習で芸術的パターンを定着
    let mut training_config = build_training_config(profile, false);
    training_config.checkpoint_dir = Some("checkpoints".to_string());
    training_config.model_name = "honrou_model".to_string();
    // 芸術的報酬は値の範囲が広いため、学習率を少し下げて安定化
    training_config.learning_rate *= 0.8;

    let mut trainer = Trainer::new(
        training_config,
        training_model,
        model_config.clone(),
        training_device,
    );
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
fn run_training_mode_honrou_gpu(profile: TrainingProfile, model_config: &AlphaZeroModelConfig) {
    println!("========================================");
    println!("  AlphaZero 翻弄 Training Mode (GPU)");
    println!("  Artistic Reward: 芸術的報酬");
    println!("========================================\n");
    println!("Training profile: {}\n", profile.as_str());

    #[cfg(feature = "wgpu")]
    {
        use burn::tensor::{Device, DeviceKind};

        let cpu_device = Device::flex();
        let gpu_device = Device::wgpu(DeviceKind::DefaultDevice);
        println!(
            "Self-play device: CPU (Flex), threads: {}",
            available_threads()
        );
        println!("Training device:  GPU (WGPU)");

        println!("Initializing AlphaZero model (artistic)...");
        println!("  - Residual blocks: {}", model_config.num_res_blocks);
        println!("  - Filters: {}", model_config.num_filters);

        let self_play_model = AlphaZeroModel::new(model_config, &cpu_device);

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

        println!("\nInitializing self-play engine (CPU inference, artistic)...");
        let mcts = MctsSearch::new(
            self_play_config.mcts_config,
            self_play_model,
            cpu_device.clone(),
        );
        let self_play = SelfPlayEngine::new(self_play_config, mcts, cpu_device.clone());

        println!("\n--- Starting Self-Play (Artistic) ---");
        let training_data = self_play.generate_data();

        println!("\n--- Starting Training on GPU (Artistic) ---");
        let training_device = gpu_device.autodiff();
        let training_model = AlphaZeroModel::new(model_config, &training_device);

        let mut training_config = build_training_config(profile, true);
        training_config.checkpoint_dir = Some("checkpoints".to_string());
        training_config.model_name = "honrou_model".to_string();
        training_config.learning_rate *= 0.8;

        let mut trainer = Trainer::new(
            training_config,
            training_model,
            model_config.clone(),
            training_device,
        );
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
        run_training_mode_honrou(profile, model_config);
    }
}

/// 自己対戦と学習の実測速度を表示する（ボトルネック確認用）。
///
/// 使い方: `cargo run --release -- --bench [ゲーム数] [--sims N] [--model light|standard]`
fn run_bench_mode(args: &[String]) {
    use burn::tensor::Device;
    use std::time::Instant;

    let position = |flag: &str| args.iter().position(|arg| arg == flag);

    let games: usize = position("--bench")
        .and_then(|index| args.get(index + 1))
        .and_then(|value| value.parse().ok())
        .unwrap_or(2);
    let simulations: usize = position("--sims")
        .and_then(|index| args.get(index + 1))
        .and_then(|value| value.parse().ok())
        .unwrap_or(200);
    let eval_batch: usize = position("--batch")
        .and_then(|index| args.get(index + 1))
        .and_then(|value| value.parse().ok())
        .unwrap_or(8);

    let model_config = model_config_from_args(args);
    let device = Device::flex();

    println!("========================================");
    println!("  Benchmark (CPU / Flex)");
    println!("========================================");
    println!("threads       : {}", available_threads());
    println!(
        "model         : {} blocks x {} filters",
        model_config.num_res_blocks, model_config.num_filters
    );
    println!(
        "self-play     : {} games, {} simulations/move",
        games, simulations
    );
    println!("eval batch    : {}", eval_batch);

    // NN フォワードのバッチ特性を測る。バッチを増やしても時間が増えないなら
    // 1 演算あたりのオーバーヘッド支配で、MCTS のバッチ評価は効かないと判断できる。
    {
        use crate::game::BOARD_SIZE;
        use burn::tensor::Tensor;

        println!("\n--- forward pass scaling ---");
        let probe_model = AlphaZeroModel::new(&model_config, &device);
        for batch in [1usize, 4, 8, 32] {
            let input = Tensor::<4>::zeros([batch, 3, BOARD_SIZE, BOARD_SIZE], &device);
            let _ = probe_model.forward(input.clone()); // ウォームアップ
            let start = Instant::now();
            for _ in 0..5 {
                let _ = probe_model.forward(input.clone());
            }
            let ms = start.elapsed().as_secs_f64() * 1000.0 / 5.0;
            println!(
                "batch {batch:>3}: {ms:>8.2} ms/forward ({:.2} ms/sample)",
                ms / batch as f64
            );
        }
    }

    let model = AlphaZeroModel::new(&model_config, &device);
    let self_play_config = SelfPlayConfig {
        num_games: games,
        num_mixed_games: 0,
        checkpoint_interval: 1,
        mcts_config: MctsConfig {
            num_simulations: simulations,
            max_search_time_ms: None,
            min_simulations_before_stop: simulations,
            early_stop_visit_ratio: None,
            eval_batch_size: eval_batch,
            ..Default::default()
        },
        ..Default::default()
    };

    let mcts = MctsSearch::new(self_play_config.mcts_config, model, device.clone());
    let engine = SelfPlayEngine::new(self_play_config, mcts, device.clone());

    let start = Instant::now();
    let training_data = engine.generate_data();
    let elapsed = start.elapsed().as_secs_f64();

    // 1 局あたりの手数は「例数 / 8（対称拡張）」で概算できる
    let moves = training_data.len() as f64 / 8.0;
    let simulations_total = moves * simulations as f64;

    println!("\n--- self-play ---");
    println!("time          : {:.1}s", elapsed);
    println!("games/sec     : {:.3}", games as f64 / elapsed);
    println!(
        "examples      : {} ({:.0} ex/sec)",
        training_data.len(),
        training_data.len() as f64 / elapsed
    );
    println!(
        "mcts sims/sec : {:.0} (概算, {} moves total)",
        simulations_total / elapsed,
        moves as usize
    );

    let training_device = device.clone().autodiff();
    let training_model = AlphaZeroModel::new(&model_config, &training_device);
    let training_config = TrainingConfig {
        num_epochs: 1,
        steps_per_epoch: usize::MAX,
        checkpoint_dir: None,
        ..Default::default()
    };
    let mut trainer = Trainer::new(
        training_config,
        training_model,
        model_config.clone(),
        training_device,
    );

    // 初回はカーネル初期化を含むため、2 回目のエポックを計測する
    trainer.train(&training_data);
    let start = Instant::now();
    trainer.train(&training_data);
    let elapsed = start.elapsed().as_secs_f64();

    println!("\n--- training (1 epoch, 全バッチ / ウォームアップ後) ---");
    println!("time          : {:.1}s", elapsed);
    println!(
        "examples/sec  : {:.0}",
        training_data.len() as f64 / elapsed
    );
}

/// GPU (Vulkan / WebGPU) の速度を計測する。`--features wgpu` が必要。
///
/// 使い方: `cargo run --release --features wgpu -- --bench-gpu [--model light|standard]`
#[allow(unexpected_cfgs)]
fn run_bench_gpu_mode(args: &[String]) {
    #[cfg(feature = "wgpu")]
    {
        use crate::game::BOARD_SIZE;
        use burn::tensor::{Device, DeviceKind, Tensor};
        use std::time::Instant;

        let model_config = model_config_from_args(args);

        println!("========================================");
        println!("  Benchmark (GPU / wgpu = Vulkan or DX12)");
        println!("========================================");
        println!(
            "model         : {} blocks x {} filters",
            model_config.num_res_blocks, model_config.num_filters
        );

        let start = Instant::now();
        let device = Device::wgpu(DeviceKind::DefaultDevice);
        println!("device init   : {:.2?}", start.elapsed());
        println!("device        : {device:?}");

        // 順伝播（初回はカーネル生成が走るのでウォームアップしてから計測）
        let model = AlphaZeroModel::new(&model_config, &device);
        println!("\n--- forward pass ---");
        for batch in [1usize, 32] {
            let input = Tensor::<4>::zeros([batch, 3, BOARD_SIZE, BOARD_SIZE], &device);
            let _ = model.forward(input.clone());
            let start = Instant::now();
            for _ in 0..5 {
                let _ = model.forward(input.clone());
            }
            let ms = start.elapsed().as_secs_f64() * 1000.0 / 5.0;
            println!(
                "batch {batch:>3}: {ms:>8.2} ms/forward ({:.2} ms/sample)",
                ms / batch as f64
            );
        }

        // 学習 1 エポック（自己対戦は CPU で行うため、ここではダミーデータを使う）
        let examples: Vec<crate::ai::alphazero::TrainingExample> = (0..512)
            .map(|_| crate::ai::alphazero::TrainingExample {
                board_tensor: vec![0.0; 3 * BOARD_SIZE * BOARD_SIZE],
                target_policy: vec![
                    1.0 / (BOARD_SIZE * BOARD_SIZE) as f32;
                    BOARD_SIZE * BOARD_SIZE
                ],
                value: 0.0,
            })
            .collect();

        let training_device = device.clone().autodiff();
        let training_model = AlphaZeroModel::new(&model_config, &training_device);
        let training_config = TrainingConfig {
            num_epochs: 1,
            batch_size: 256,
            steps_per_epoch: usize::MAX,
            checkpoint_dir: None,
            ..Default::default()
        };
        let mut trainer = Trainer::new(
            training_config,
            training_model,
            model_config.clone(),
            training_device,
        );

        println!("\n--- training (1 epoch, {} examples) ---", examples.len());
        let start = Instant::now();
        trainer.train(&examples);
        let first = start.elapsed().as_secs_f64();

        let start = Instant::now();
        trainer.train(&examples);
        let elapsed = start.elapsed().as_secs_f64();
        println!("1st epoch (warmup) : {first:.1}s");
        println!(
            "2nd epoch          : {elapsed:.1}s ({:.0} examples/sec)",
            examples.len() as f64 / elapsed
        );
    }

    #[cfg(not(feature = "wgpu"))]
    {
        let _ = args;
        println!("GPU support not enabled. Compile with --features wgpu");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_known_font_containers() {
        assert!(looks_like_font(&[0x00, 0x01, 0x00, 0x00, 0xFF]));
        assert!(looks_like_font(b"OTTO____"));
        assert!(looks_like_font(b"true____"));
        assert!(looks_like_font(b"ttcf____"));
    }

    #[test]
    fn rejects_non_font_data() {
        assert!(!looks_like_font(b"<!DOCTYPE html>"));
        assert!(!looks_like_font(b"\x89PNG"));
        assert!(!looks_like_font(b""));
        assert!(!looks_like_font(b"tt"));
    }

    #[test]
    fn font_candidates_are_ordered_and_non_empty() {
        let candidates = japanese_font_candidates();
        assert!(candidates.len() > 1);
        // リポジトリ同梱フォントが最優先であること
        assert_eq!(
            candidates[0],
            std::path::PathBuf::from("fonts/BIZ_UDPGothic/BIZUDPGothic-Regular.ttf")
        );
    }

    #[test]
    fn training_profile_defaults_to_high_quality() {
        let args = vec!["neko-ai".to_string(), "--training".to_string()];
        assert_eq!(
            TrainingProfile::from_args(&args),
            TrainingProfile::HighQuality
        );

        let args = vec!["neko-ai".to_string(), "--fast".to_string()];
        assert_eq!(TrainingProfile::from_args(&args), TrainingProfile::Fast);
    }

    #[test]
    fn self_play_config_is_consistent_across_profiles() {
        for profile in [
            TrainingProfile::Fast,
            TrainingProfile::Balanced,
            TrainingProfile::HighQuality,
        ] {
            for is_gpu in [false, true] {
                let config = build_self_play_config(profile, is_gpu);
                assert!(config.num_games > 0);
                assert!(config.mcts_config.num_simulations > 0);
                assert_eq!(config.reward_mode, RewardMode::Standard);
                assert!(
                    config.mcts_config.min_simulations_before_stop
                        <= config.mcts_config.num_simulations,
                    "早期終了の下限が総シミュレーション数を超えている: {:?}",
                    profile
                );

                let training = build_training_config(profile, is_gpu);
                assert!(training.num_epochs > 0);
                assert!(training.batch_size > 0);
                assert!(training.learning_rate > 0.0);
                assert_eq!(training.model_name, "best_model");
            }
        }
    }
}
