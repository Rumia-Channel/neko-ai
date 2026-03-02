# AGENTS.md - Coding Guidelines for neko-ai

## Project Overview
Rust project implementing an Othello game with GUI using eframe (egui). Uses the 2024 edition and integrates the Burn ML framework.

## Build Commands

```bash
# Build the project
cargo build

# Build release
cargo build --release

# Run the application
cargo run

# Check without building
cargo check
```

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
- **Modules**: snake_case (e.g., `game`, `othello`)
- **Generic parameters**: Single uppercase letters (e.g., `T`, `G`)

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
- Document panics, errors, and safety invariants
- Include examples in doc comments for complex functions

### Module Structure
```
src/
├── main.rs          # Entry point, app initialization
├── app.rs           # GUI application implementation
└── game/
    ├── mod.rs       # Game trait, shared types
    └── othello.rs   # Specific game implementations
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
- `burn` (0.20.1): ML framework with cpu/webgpu/vulkan/ndarray features
- `eframe` (0.33.3): GUI framework

## Project-specific Notes

- Game logic is trait-based (`Game` trait in `game/mod.rs`)
- GUI uses immediate-mode rendering via egui
- Japanese font is embedded at `fonts/GDhwGoJA-OTF112b2.otf`
- Board size is fixed at 8x8 (`const BOARD_SIZE: usize = 8`)
