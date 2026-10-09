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
cargo run --release -- --training-fast      # 約 20 分（このマシン実測 22.5 分）
cargo run --release -- --training           # --high 相当（数時間）
cargo run --release -- --training-balanced  # 中間
```

> **必ず `--release` で実行してください**（デバッグビルドは 10 倍以上遅くなります）。
> 既定は CPU でも回る軽量モデル（4 blocks × 64 filters, 約 50 万パラメータ）です。
> 大きいモデルを使う場合は `--model standard`（10 × 256, 約 4 倍重い）。

#### GPU モード（Vulkan / WebGPU）
```bash
cargo run --release --features wgpu -- --training-gpu
cargo run --release --features wgpu -- --bench-gpu      # GPU の速度計測
```

> **Vulkan / DX12 はビルド・動作確認済み**（NVIDIA / AMD / Intel いずれの Vulkan でも動作）。
> `wgpu-hal 30` は `windows 0.62` を要求しますが、依存の `gpu-allocator 0.28` は単独では
> `windows 0.61` に解決されて型が食い違います。`Cargo.toml` で `windows = "0.62"` を直接指定し、
> `Cargo.lock` も 0.62.2 に統一することで解消しています。
> 自己対戦は CPU（Flex）、学習のみ GPU で行う構成です（バッチ1の GPU 推論は転送が重いため）。

#### 速度計測
```bash
cargo run --release -- --bench 32 --sims 120            # 自己対戦 + 学習 1 エポック
cargo run --release -- --bench 32 --sims 120 --model standard
```

実測値（i7-1355U / 12 スレッド / 軽量モデル, release ビルド）:

| 項目 | 実測 |
|------|------|
| MCTS 探索 | 約 2,500 simulations/秒（12 ワーカー並列） |
| 自己対戦 | 約 0.34 games/秒（120 sims/手） |
| 学習 | 約 390 examples/秒 |
| `--training-fast` 完走 | 22.5 分（自己対戦 2.3 分 + 学習 20 分） |

> 学習は CPU では「examples/秒 × エポック数」に比例します。時間を短縮したい場合は
> `--fast`、`--model light`（既定）、あるいはエポック数を減らしてください。

## AI 難易度

| 難易度 | アルゴリズム | 特徴 |
|--------|-------------|------|
| 簡単 | Monte Carlo | 3手先読み、石数最大化 |
| 普通 | Monte Carlo | 5手先読み、シミュレーション増加 |
| 少し難しい | Restrictive | 相手の選択肢を最小化する戦略 |
| 難しい | AlphaZero | `checkpoints/best_model.bpk` を読み込み、NN + MCTS |
| 翻弄 | AlphaZero | `checkpoints/honrou_model.bpk`（芸術的報酬で学習）。未学習時は HonrouAi にフォールバック |

> モデル未学習（`checkpoints/*.bpk` が無い）場合、難しい/翻弄は HonrouAi にフォールバックします。
> モデル構成（blocks/filters）は `.bpk.config` サイドカーに保存され、読み込み時に自動で復元されます。

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
