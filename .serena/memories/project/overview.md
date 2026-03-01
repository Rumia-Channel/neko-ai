# プロジェクト概要

## 目的
Rustでegui(eframe)とburnを使用してオセロゲームとAI対戦システムを構築する。

## 技術スタック
- **言語**: Rust (Edition 2024)
- **GUIフレームワーク**: egui / eframe 0.33.3
- **機械学習フレームワーク**: burn 0.20.1
  - Features: cpu, webgpu, vulkan, ndarray

## プロジェクト構造
```
neko-ai/
├── Cargo.toml
├── src/
│   └── main.rs
└── target/
```

## コマンド
- `cargo run` - アプリケーションの実行
- `cargo build` - ビルド
- `cargo build --release` - リリースビルド
- `cargo check` - コンパイルチェック
- `cargo add <crate>` - 依存関係の追加
- `cargo fmt` - コードフォーマット
- `cargo clippy` - リンタ/静的解析

## Windows固有のコマンド
- `dir` - ファイル一覧
- `cd` - ディレクトリ変更
