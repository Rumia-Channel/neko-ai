mod ai;
mod app;
mod game;

use ai::alphazero::{AlphaZeroModel, AlphaZeroModelConfig, SelfPlayEngine, Trainer};
use app::GameApp;
use burn::backend::ndarray::NdArrayDevice;
use eframe::egui;
use std::env;

fn main() {
    // Parse command line arguments
    let args: Vec<String> = env::args().collect();

    if args.len() > 1 && args[1] == "--training" {
        // Run training mode
        run_training_mode();
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
    println!("  AlphaZero Training Mode for Othello");
    println!("========================================\n");

    // Setup device
    let device = NdArrayDevice::default();
    println!("Using device: NdArray (CPU)");

    // Create model with default configuration
    let model_config = AlphaZeroModelConfig::new();

    // Initialize model
    println!("Initializing AlphaZero model...");
    println!("  - Residual blocks: {}", model_config.num_res_blocks);
    println!("  - Filters: {}", model_config.num_filters);

    type Backend = burn::backend::ndarray::NdArray;
    let model = AlphaZeroModel::<Backend>::new(&model_config, &device);

    // Create self-play engine
    println!("\nInitializing self-play engine...");
    let self_play = SelfPlayEngine::new(Default::default(), model, device);

    // Generate training data
    println!("\n--- Starting Self-Play ---");
    let training_data = self_play.generate_data();

    // Create trainer with default configuration
    println!("\n--- Starting Training ---");
    let training_model: AlphaZeroModel<Backend> = AlphaZeroModel::new(&model_config, &device);
    let mut trainer = Trainer::new(Default::default(), training_model, device);
    trainer.train(&training_data);

    println!("\n========================================");
    println!("  Training Complete!");
    println!("========================================");
    println!("\nTo play against the trained AI, run:");
    println!("  cargo run");
}
