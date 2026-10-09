# Neko AI - オセロゲーム with AlphaZero

Rust で実装されたオセロゲーム GUI アプリケーション。AlphaZero 機械学習 AI を搭載。

## 機能

- **GUI モード**: 直感的なインターフェースで対戦
- **AI 対戦**: 5段階の難易度（簡単、普通、少し難しい、難しい、翻弄）
- **AlphaZero**: ディープラーニング + MCTS による最強 AI
- **人間対人間**: 2人プレイ対応

## 必要条件

- Rust 1.95+
- （GPU 学習時）Vulkan/WebGPU 対応 GPU

> **フォントについて**: 日本語表示用の BIZ UDPGothic はリポジトリに含まれていません
> （`*.ttf` は .gitignore 対象）。アプリは起動時に
> `fonts/BIZ_UDPGothic/BIZUDPGothic-Regular.ttf` を探し、無ければ OS 標準の日本語
> フォント（Windows なら BIZ UDGothic / メイリオ等）を使います。
> 同梱したい場合は上記パスにフォントを配置してください。

> **学習済みモデルについて**: モデルは burnpack 形式（`.bpk`）で
> `checkpoints/best_model.bpk`（難易度「難しい」）と `checkpoints/honrou_model.bpk`
> （「翻弄」）に保存されます。未学習の場合は自動的に HonrouAi へフォールバックします。

## インストール

```bash
# リポジトリをクローン
git clone <repository-url>
cd neko-ai

# ビルド
cargo build --release
```

## 使い方

### GUI モード（通常プレイ）

```bash
cargo run
```

### AlphaZero 学習モード

#### CPU モード
```bash
cargo run -- --training
```

#### GPU モード（高速）
```bash
# wgpu フィーチャーが必要
cargo run --features wgpu -- --training-gpu
```

> **⚠ GPU 学習は現在ビルドできません（上流バグ）**: burn 0.22 の GPU バックエンドは
> `cubecl-wgpu 0.11` 経由で `wgpu 30.0.1` を要求しますが、この版は Windows で
> コンパイルできません（`wgpu-hal` が `windows 0.62` を要求する一方、依存の
> `gpu-allocator 0.28` は `windows <= 0.62` 指定で 0.61 に解決されるため型が不一致）。
> `wgpu 30.0.1` が最新のため現時点で回避策はありません。
> 自己対戦・学習とも CPU（Flex バックエンド）で実行してください
> （`--features wgpu` を付けなければ CPU のみでビルドされます）。

## AI 難易度

| 難易度 | アルゴリズム | 特徴 |
|--------|-------------|------|
| 簡単 | Monte Carlo | 3手先読み、石数最大化 |
| 普通 | Monte Carlo | 5手先読み、シミュレーション増加 |
| 少し難しい | Restrictive | 相手の選択肢を最小化する戦略 |
| 難しい | AlphaZero | ニューラルネットワーク + MCTS |
| 翻弄 | Honrou | 最上級、深い探索（7手先） |

## 操作方法

- **マウスクリック**: 石を置く
- **新規ゲーム**: ゲームをリセット
- **AI の手番**: 自動的に AI が手を選択

## プロジェクト構成

```
src/
├── main.rs              # エントリーポイント
├── app.rs               # GUI アプリケーション
├── game/
│   ├── mod.rs           # ゲームトレイト
│   └── othello.rs       # オセロ実装
└── ai/
    ├── mod.rs           # AI インターフェース
    ├── alphazero/       # AlphaZero 実装
    ├── honrou.rs        # 翻弄 AI
    ├── monte_carlo.rs   # モンテカルロ AI
    └── restrictive.rs   # 制限的 AI
```

## ライセンス

MIT License
