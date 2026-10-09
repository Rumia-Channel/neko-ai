# AGENTS.md - Coding Guidelines for neko-ai

## Project Overview
Rust project implementing an Othello game with GUI using eframe (egui). Uses the 2024 edition and integrates the Burn ML framework with AlphaZero AI.

## Build Commands

```bash
# Build the project
cargo build

# Build release
cargo build --release

# Run the GUI application
cargo run

# Run training mode (CPU)
cargo run -- --training

# Run training mode (GPU - requires wgpu feature)
cargo run --features wgpu -- --training-gpu

# Run training mode (Honrou / artistic reward)
cargo run -- --training-honrou

# Check without building
cargo check

# Check with all features
cargo check --all-features
```

NOTE: The `wgpu` feature enables the GPU (CubeCL/wgpu) backend used by
`--training-gpu`. Without it, the binary builds CPU-only (Flex) and the GPU
training modes fall back to CPU with a message.

WARNING: the `wgpu` feature (and therefore `cargo check --all-features`) does **not**
compile on Windows right now: `cubecl-wgpu 0.11` requires `wgpu 30.0.1`, whose
`wgpu-hal` needs `windows 0.62` while its `gpu-allocator 0.28` dependency is capped at
`windows <= 0.62` and resolves to 0.61. `wgpu 30.0.1` is the latest release, so there
is no workaround except waiting for an upstream fix.

## Test Commands

```bash
# Run all tests
cargo test

# Run a specific test by name
cargo test <test_name>

# Run tests with output
cargo test -- --nocapture

# Run tests for a specific module
cargo test game::

# Run tests for AI module
cargo test ai::
```

## Lint Commands

```bash
# Run clippy (required before committing)
cargo clippy -- -D warnings

# Run clippy with all features
cargo clippy --all-features -- -D warnings

# Format code (required before committing)
cargo fmt

# Check formatting without modifying
cargo fmt -- --check
```

## Code Style Guidelines

### Formatting
- Use standard Rust formatting via `cargo fmt` (no custom rustfmt.toml)
- Maximum line length: 100 characters (soft limit)
- 4 spaces for indentation
- Trailing commas in multi-line structures

### Naming Conventions
- **Types/Structs/Enums/Traits**: PascalCase (e.g., `OthelloGame`, `Cell`, `Player`)
- **Functions/Methods/Variables**: snake_case (e.g., `make_move`, `black_count`)
- **Constants/Static**: SCREAMING_SNAKE_CASE (e.g., `BOARD_SIZE`, `CELL_SIZE`)
- **Modules**: snake_case (e.g., `game`, `othello`, `alphazero`)
- **Generic parameters**: Single uppercase letters (e.g., `T`, `G`, `B`)

### Imports
- Group imports: std, external crates, then crate-local
- Use `use super::` for parent module imports
- Prefer explicit imports over glob imports (`*`)
- Import order example:
  ```rust
  use std::fmt::Debug;

  use eframe::egui;

  use crate::game::{Cell, Game};
  ```

### Type Usage
- Prefer explicit types for public API
- Use `#[derive(...)]` for standard traits: `Debug`, `Clone`, `Copy`, `PartialEq`, `Default`
- Use `Box<dyn Trait>` for trait objects when needed
- Use `const` for compile-time constants
- Use arrays for fixed-size collections: `[[Cell; BOARD_SIZE]; BOARD_SIZE]`

### Error Handling
- Use `Result<T, E>` for fallible operations
- Use `Option<T>` for nullable values
- Prefer early returns: `if !condition { return false; }`
- Use `unwrap()` only in tests or when panic is truly unreachable
- Propagate errors with `?` operator

### Traits
- Use trait bounds for generic constraints: `Game: Send + Debug`
- Provide default implementations when appropriate
- Document trait contracts in doc comments

### Documentation
- Use `///` for public API documentation
- Use `//!` for module-level documentation
- Document panics, errors, and safety invariants
- Include examples in doc comments for complex functions

### Module Structure
```
src/
├── main.rs              # Entry point, CLI argument parsing
├── app.rs               # GUI application implementation
├── game/
│   ├── mod.rs           # Game trait, Cell/Player enums, BOARD_SIZE
│   └── othello.rs       # OthelloGame implementation
└── ai/
    ├── mod.rs           # AiPlayer trait, AI difficulty levels
    ├── alphazero/       # AlphaZero ML implementation
    │   ├── mod.rs
    │   ├── model.rs     # Neural network model
    │   ├── mcts.rs      # Monte Carlo Tree Search
    │   ├── self_play.rs # Self-play data generation
    │   ├── training.rs  # Model training
    │   ├── artistic.rs  # Artistic reward for Honrou (翻弄) training
    │   ├── player.rs    # AlphaZeroPlayer: AiPlayer adapter
    │   └── tensor_utils.rs # Board -> tensor conversion
    ├── honrou.rs        # Advanced AI strategy
    ├── monte_carlo.rs   # Monte Carlo AI
    └── restrictive.rs   # Restrictive strategy AI
```

## Pre-commit Checklist

Run these before committing:
```bash
cargo fmt
cargo clippy -- -D warnings
cargo test
```

## Dependencies

Key external crates:
- `burn` (0.22.0): ML framework. Enabled features: `std`, `optim` (implies `autodiff`),
  `flex`, `store`. `default-features = false`.
  - CPU execution uses the pure-Rust **Flex** backend (`Device::flex()`); the `ndarray`
    backend is deprecated in 0.22.
  - The `cpu` (CubeCL CPU) feature is intentionally **not** enabled: it pulls
    cubecl-cpu -> tracel-llvm -> liblzma-sys and builds an LLVM bundle.
  - GPU backends (`webgpu` + `vulkan`) are enabled by this crate's own `wgpu` feature.
- `eframe` (0.36.2): GUI framework, `default-features = false` with `glow` renderer.
  - The default `wgpu` renderer is avoided because wgpu 30.0.1 does not compile on
    Windows: `wgpu-hal` requires `windows 0.62` while its `gpu-allocator 0.28` dependency
    is capped at `windows <= 0.62` (resolves to 0.61), so the types mismatch (LNK/E0277).
- `rand` (0.10.0): Random number generation

### burn 0.22 API notes (migrated from 0.20)

- Tensors are `Tensor<D>` / `Tensor<D, Int>`: no backend type parameter.
- Devices are runtime values: `Device::flex()`, `Device::wgpu(DeviceKind::DefaultDevice)`,
  `Device::default()`. `Device::default()` panics if no backend feature is enabled.
- Autodiff is a runtime device context: `let device = Device::flex().autodiff();`
  Build the model *after* enabling it. `Autodiff<B>` / `AutodiffBackend` no longer exist.
- Models derive `Module` without generics; optimizer is the concrete `ModuleOptimizer`
  from `AdamConfig::new().init()`, stepped with `optimizer.step(lr, model, grads)`.
- Checkpoints use the burnpack format (`.bpk`) via `model.into_record().save(path)` and
  `ModuleRecord::load(path)` + `model.load_record(record)`. Old `.mpk` files (burn 0.20)
  cannot be loaded — models must be retrained.
- `PaddingConfig2d::Explicit(top, left, bottom, right)` now takes four values.
- `eframe::App` requires `fn ui(&mut self, ui: &mut egui::Ui, frame: &mut Frame)` and offers
  `fn logic(&mut self, ctx, frame)` for non-drawing per-frame work (used for AI turns).

## Project-specific Notes

- Game logic is trait-based (`Game` trait in `game/mod.rs`)
- GUI uses immediate-mode rendering via egui; AI turns run on a background thread and are
  applied from `GameApp::poll_ai_job` (never block the UI thread with `choose_move`)
- Japanese font is loaded **at runtime** (`load_japanese_font` in `main.rs`): it searches
  `fonts/BIZ_UDPGothic/BIZUDPGothic-Regular.ttf` first, then OS Japanese fonts.
  The font file is not committed (`*.ttf` is gitignored), so it must never be `include_bytes!`-ed
- Board size is fixed at 8x8 (`const BOARD_SIZE: usize = 8`)
- AI difficulty levels: Easy, Medium, SlightlyHard, Hard, Honrou
- Command-line args: `--training` (CPU mode), `--training-gpu` (GPU mode)
