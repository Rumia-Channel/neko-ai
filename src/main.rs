mod ai;
mod app;
mod game;

use ai::alphazero::{
    AlphaZeroModel, AlphaZeroModelConfig, MctsConfig, MctsSearch, SelfPlayConfig, SelfPlayEngine,
    Trainer, TrainingConfig,
};
use app::GameApp;
use eframe::egui;
use std::env;

fn main() {
    // Parse command line arguments
    let args: Vec<String> = env::args().collect();

    if args.len() > 1 && args[1] == "--training" {
        // Run training mode
        run_training_mode();
    } else if args.len() > 1 && args[1] == "--training-gpu" {
        // Run training mode with GPU
        run_training_mode_gpu();
    } else {
        // Run GUI mode
        if let Err(e) = run_gui_mode() {
            eprintln!("Error running GUI: {}", e);
            std::process::exit(1);
        }
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

fn run_training_mode() {
    println!("========================================");
    println!("  AlphaZero Training Mode (CPU)");
    println!("========================================\n");

    use burn::backend::ndarray::NdArrayDevice;

    // Setup device
    let device = NdArrayDevice::default();
    println!("Using device: CPU (NdArray)");

    // Create model configuration
    let model_config = AlphaZeroModelConfig::new();
    println!("Initializing AlphaZero model...");
    println!("  - Residual blocks: {}", model_config.num_res_blocks);
    println!("  - Filters: {}", model_config.num_filters);

    type Backend = burn::backend::ndarray::NdArray;
    let model = AlphaZeroModel::<Backend>::new(&model_config, &device);

    // Configure self-play
    let self_play_config = SelfPlayConfig {
        num_games: 100,
        temperature: 1.0,
        temp_threshold: 30,
        temp_after_threshold: 0.0,
        checkpoint_interval: 10,
        mcts_config: MctsConfig {
            num_simulations: 800,
            dirichlet_alpha: 0.3,
            dirichlet_weight: 0.25,
            temperature: 1.0,
            c_puct: 1.0,
        },
    };

    println!("\nSelf-play configuration:");
    println!("  - Games: {}", self_play_config.num_games);
    println!(
        "  - MCTS simulations: {}",
        self_play_config.mcts_config.num_simulations
    );

    // Create self-play engine
    println!("\nInitializing self-play engine...");
    let mcts = MctsSearch::new(self_play_config.mcts_config, model, device);
    let self_play = SelfPlayEngine::new(self_play_config, mcts, device);

    // Generate training data
    println!("\n--- Starting Self-Play ---");
    let training_data = self_play.generate_data();

    // Training
    println!("\n--- Starting Training ---");
    let training_model = AlphaZeroModel::<Backend>::new(&model_config, &device);
    let training_config = TrainingConfig {
        num_epochs: 50,
        batch_size: 256,
        learning_rate: 0.001,
        lr_decay: 0.1,
        lr_decay_epochs: 15,
        steps_per_epoch: 500,
        validation_split: 0.1,
        checkpoint_dir: Some("checkpoints".to_string()),
        save_every: 5,
    };

    let mut trainer = Trainer::new(training_config, training_model, device);
    trainer.train(&training_data);

    println!("\n========================================");
    println!("  Training Complete!");
    println!("========================================");
}

#[allow(unexpected_cfgs)]
fn run_training_mode_gpu() {
    println!("========================================");
    println!("  AlphaZero Training Mode (GPU)");
    println!("========================================\n");

    #[cfg(feature = "wgpu")]
    {
        use burn::backend::wgpu::WgpuDevice;

        // Setup GPU device
        let device = WgpuDevice::default();
        println!("Using device: GPU (WGPU)");

        // Create model configuration
        let model_config = AlphaZeroModelConfig::new();
        println!("Initializing AlphaZero model...");
        println!("  - Residual blocks: {}", model_config.num_res_blocks);
        println!("  - Filters: {}", model_config.num_filters);

        type Backend = burn::backend::wgpu::Wgpu;
        let model = AlphaZeroModel::<Backend>::new(&model_config, &device);

        // Configure self-play with GPU
        let self_play_config = SelfPlayConfig {
            num_games: 1000, // More games with GPU
            temperature: 1.0,
            temp_threshold: 30,
            temp_after_threshold: 0.0,
            checkpoint_interval: 100,
            mcts_config: MctsConfig {
                num_simulations: 1600, // More simulations with GPU
                dirichlet_alpha: 0.3,
                dirichlet_weight: 0.25,
                temperature: 1.0,
                c_puct: 1.0,
            },
        };

        println!("\nSelf-play configuration:");
        println!("  - Games: {}", self_play_config.num_games);
        println!(
            "  - MCTS simulations: {}",
            self_play_config.mcts_config.num_simulations
        );

        // Create self-play engine
        println!("\nInitializing self-play engine with GPU...");
    let mcts = MctsSearch::new(self_play_config.mcts_config, model, device);
        let self_play = SelfPlayEngine::new(self_play_config, mcts, device);

        // Generate training data
        println!("\n--- Starting Self-Play ---");
        let training_data = self_play.generate_data();

        // Training
        println!("\n--- Starting Training on GPU ---");
        let training_model = AlphaZeroModel::<Backend>::new(&model_config, &device);
        let training_config = TrainingConfig {
            num_epochs: 100,
            batch_size: 512,
            learning_rate: 0.001,
            lr_decay: 0.1,
            lr_decay_epochs: 30,
            steps_per_epoch: 1000,
            validation_split: 0.1,
            checkpoint_dir: Some("checkpoints".to_string()),
            save_every: 10,
        };

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
        run_training_mode();
    }
}
