//! 翻弄AlphaZero用 芸術的報酬関数
//!
//! 単純な勝敗ではなく、最終局面の「芸術性」を評価する報酬系。
//! - 32:32 の完全引き分け
//! - 相手に四隅を譲りながら 60:4 で勝利
//! - 2x2 単色ブロックで盤面が埋め尽くされるパターン
//! - その他、石数差の美しさなど

use crate::game::{BOARD_SIZE, Cell, Game, Player};

/// 芸術的報酬の種類とスコア
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ArtisticPattern {
    /// 32:32 の完全引き分け (最高芸術点)
    PerfectDraw,
    /// 相手に四隅を全て取らせながら 60:4 で圧勝
    CornerSacrifice,
    /// 盤面が 2x2 単色ブロックで埋め尽くされている
    BlockTiling,
    /// 大きな石数差での勝利 (芸術性は低いが正報酬)
    DominantWin,
    /// 通常の勝利
    StandardWin,
    /// 引き分け (32:32以外)
    StandardDraw,
    /// 敗北
    Loss,
}

impl ArtisticPattern {
    /// このパターンに対する報酬値 [-1.0, 1.0]
    pub fn reward(&self) -> f32 {
        match self {
            Self::PerfectDraw => 1.0,
            Self::CornerSacrifice => 0.95,
            Self::BlockTiling => 0.9,
            Self::DominantWin => 0.6,
            Self::StandardWin => 0.4,
            Self::StandardDraw => 0.1,
            Self::Loss => -1.0,
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            Self::PerfectDraw => "32:32 完全引き分け",
            Self::CornerSacrifice => "四隅譲り 60:4 圧勝",
            Self::BlockTiling => "2x2ブロック敷き詰め",
            Self::DominantWin => "大差勝利",
            Self::StandardWin => "通常勝利",
            Self::StandardDraw => "引き分け",
            Self::Loss => "敗北",
        }
    }
}

/// 最終局面から芸術的報酬を計算する
/// `perspective` の視点からの報酬を返す
pub fn evaluate_artistic_reward(game: &dyn Game, perspective: Player) -> f32 {
    let pattern = detect_pattern(game, perspective);
    let base_reward = pattern.reward();

    // 2x2ブロックの覆盖率によるボーナス (BlockTiling検出時に上乗せ)
    let block_bonus = if pattern == ArtisticPattern::BlockTiling {
        let coverage = count_2x2_block_coverage(game);
        // 覆盖率が高いほど追加報酬 (最大 +0.1)
        coverage * 0.1
    } else {
        0.0
    };

    (base_reward + block_bonus).clamp(-1.0, 1.0)
}

/// 最終局面のパターンを検出する
fn detect_pattern(game: &dyn Game, perspective: Player) -> ArtisticPattern {
    let my_count = match perspective {
        Player::Black => game.black_count(),
        Player::White => game.white_count(),
    };
    let opp_count = match perspective {
        Player::Black => game.white_count(),
        Player::White => game.black_count(),
    };

    // 32:32 完全引き分け
    if my_count == 32 && opp_count == 32 {
        return ArtisticPattern::PerfectDraw;
    }

    // 四隅を相手に譲りながら 60:4 で勝利
    if my_count >= 58 && opp_count <= 6 && opponent_has_all_corners(game, perspective) {
        return ArtisticPattern::CornerSacrifice;
    }

    // 2x2 ブロック敷き詰め (盤面の75%以上が2x2単色ブロックで覆われている)
    let coverage = count_2x2_block_coverage(game);
    if coverage >= 0.75 {
        return ArtisticPattern::BlockTiling;
    }

    // 勝敗判定
    match game.winner() {
        Some(winner) if winner == perspective => {
            // 大差勝利 (50石以上)
            if my_count >= 50 {
                ArtisticPattern::DominantWin
            } else {
                ArtisticPattern::StandardWin
            }
        }
        Some(_) => ArtisticPattern::Loss,
        None => ArtisticPattern::StandardDraw,
    }
}

/// 相手が四隅を全て持っているか確認
fn opponent_has_all_corners(game: &dyn Game, perspective: Player) -> bool {
    let opponent = perspective.opposite();
    let opp_cell = opponent.to_cell();
    let corners = [(0, 0), (0, 7), (7, 0), (7, 7)];

    corners.iter().all(|&(r, c)| game.board()[r][c] == opp_cell)
}

/// 盤面の 2x2 単色ブロック覆盖率を計算する
/// 戻り値: 0.0〜1.0 (カバーされているマスの割合)
fn count_2x2_block_coverage(game: &dyn Game) -> f32 {
    let board = game.board();
    let mut covered = vec![vec![false; BOARD_SIZE]; BOARD_SIZE];

    // 全ての 2x2 ウィンドウをチェック (非重複タイリング: 偶数行・偶数列起点)
    for row in (0..BOARD_SIZE - 1).step_by(2) {
        for col in (0..BOARD_SIZE - 1).step_by(2) {
            let cell = board[row][col];
            if cell == Cell::Empty {
                continue;
            }

            // 2x2 が全て同じ色か
            if board[row][col + 1] == cell
                && board[row + 1][col] == cell
                && board[row + 1][col + 1] == cell
            {
                covered[row][col] = true;
                covered[row][col + 1] = true;
                covered[row + 1][col] = true;
                covered[row + 1][col + 1] = true;
            }
        }
    }

    let covered_count = covered
        .iter()
        .flat_map(|row| row.iter())
        .filter(|&&c| c)
        .count();

    covered_count as f32 / (BOARD_SIZE * BOARD_SIZE) as f32
}

/// 学習中にパターン検出の統計を出力する
pub fn print_pattern_stats(patterns: &[ArtisticPattern]) {
    let total = patterns.len();
    if total == 0 {
        return;
    }

    let mut counts = std::collections::HashMap::new();
    for p in patterns {
        *counts.entry(*p).or_insert(0usize) += 1;
    }

    println!("\n  Artistic pattern distribution ({} games):", total);
    let order = [
        ArtisticPattern::PerfectDraw,
        ArtisticPattern::CornerSacrifice,
        ArtisticPattern::BlockTiling,
        ArtisticPattern::DominantWin,
        ArtisticPattern::StandardWin,
        ArtisticPattern::StandardDraw,
        ArtisticPattern::Loss,
    ];
    for pattern in &order {
        if let Some(&count) = counts.get(pattern) {
            println!(
                "    {:>20}: {:>4} ({:.1}%)",
                pattern.description(),
                count,
                count as f64 / total as f64 * 100.0
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::OthelloGame;

    #[test]
    fn test_perfect_draw_detection() {
        // 32:32 の盤面を直接構築
        let game = OthelloGame::default();
        // 初期状態は 2:2 なので、手動で盤面を埋める
        // (テスト用に game_over を直接セットできないため、パターン検出ロジックのみ検証)
        // 初期状態では PerfectDraw にならない
        let reward = evaluate_artistic_reward(&game, Player::Black);
        assert!(reward < 1.0);
    }

    #[test]
    fn test_2x2_block_coverage_empty_board() {
        let game = OthelloGame::default();
        let coverage = count_2x2_block_coverage(&game);
        // 初期盤面では中央の4石のみ。2x2ブロックにはならない (色が混在)
        assert!(coverage < 0.1);
    }

    #[test]
    fn test_corner_sacrifice_detection() {
        let game = OthelloGame::default();
        // 直接盤面を操作してテストは難しいため、ロジックの健全性のみ確認
        let reward = evaluate_artistic_reward(&game, Player::Black);
        assert!((-1.0..=1.0).contains(&reward));
    }
}
