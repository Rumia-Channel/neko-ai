# タスク完了時の手順

## コード変更後
1. `cargo fmt` - コードフォーマットを実行
2. `cargo clippy` - リンタチェックを実行
3. `cargo build` - コンパイル確認

## 新規依存関係追加時
`cargo add <crate>` を使用（直接Cargo.toml編集は避ける）
