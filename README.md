# Neko AI - オセロゲーム with AlphaZero

Rust で実装されたオセロゲーム GUI アプリケーション。AlphaZero 機械学習 AI を搭載。

## 機能

- **GUI モード**: 直感的なインターフェースで対戦
- **AI 対戦**: 5段階の難易度（簡単、普通、少し難しい、難しい、翻弄）
- **AlphaZero**: ディープラーニング + MCTS による最強 AI
- **人間対人間**: 2人プレイ対応

## 必要条件

- Rust 1.75+ 
- （GPU 学習時）Vulkan/WebGPU 対応 GPU

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
