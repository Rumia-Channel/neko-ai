use crate::game::{BOARD_SIZE, Game, Player};

/// AIプレイヤーのインターフェース
/// ゲーム状態を受け取り、最適な手（row, col）を返す
pub trait AiPlayer: Send + std::fmt::Debug {
    /// AIの名前を返す
    #[allow(dead_code)]
    fn name(&self) -> &'static str;

    /// 現在のゲーム状態から最適な手を選択する
    /// 有効な手がない場合はNoneを返す
    fn choose_move(&self, game: &dyn Game) -> Option<(usize, usize)>;

    /// このAIがどちらのプレイヤーとして動作するか
    #[allow(dead_code)]
    fn player(&self) -> Player;
}

/// ランダムに手を選ぶAI（テスト用・簡易的なAI）
#[derive(Debug)]
pub struct RandomAi {
    #[allow(dead_code)]
    player: Player,
}

impl RandomAi {
    pub fn new(player: Player) -> Self {
        Self { player }
    }
}

impl AiPlayer for RandomAi {
    fn name(&self) -> &'static str {
        "ランダムAI"
    }

    fn choose_move(&self, game: &dyn Game) -> Option<(usize, usize)> {
        use rand::seq::IteratorRandom;

        // 全ての有効な手を収集
        let valid_moves: Vec<(usize, usize)> = (0..BOARD_SIZE)
            .flat_map(|row| (0..BOARD_SIZE).map(move |col| (row, col)))
            .filter(|(row, col)| game.is_valid_move(*row, *col))
            .collect();

        // ランダムに1つ選ぶ
        let mut rng = rand::rng();
        valid_moves.into_iter().choose(&mut rng)
    }

    fn player(&self) -> Player {
        self.player
    }
}

/// AIの難易度レベル
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AiDifficulty {
    Easy,   // ランダム
    Medium, // 簡易的な評価関数
    Hard,   // Minimaxなど
}

impl AiDifficulty {
    pub fn as_str(&self) -> &'static str {
        match self {
            AiDifficulty::Easy => "簡単",
            AiDifficulty::Medium => "普通",
            AiDifficulty::Hard => "難しい",
        }
    }
}

/// AIプレイヤーを生成するファクトリ関数
pub fn create_ai(difficulty: AiDifficulty, player: Player) -> Box<dyn AiPlayer> {
    match difficulty {
        AiDifficulty::Easy => Box::new(RandomAi::new(player)),
        AiDifficulty::Medium => Box::new(RandomAi::new(player)), // TODO: より良いAIを実装
        AiDifficulty::Hard => Box::new(RandomAi::new(player)),   // TODO: Minimax AIを実装
    }
}

/// ゲームモード
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GameMode {
    HumanVsHuman,
    HumanVsAi,
    AiVsAi,
}

impl GameMode {
    pub fn as_str(&self) -> &'static str {
        match self {
            GameMode::HumanVsHuman => "人間 vs 人間",
            GameMode::HumanVsAi => "人間 vs AI",
            GameMode::AiVsAi => "AI vs AI",
        }
    }
}

/// ゲームプレイヤーの設定
#[derive(Debug)]
pub struct PlayerConfig {
    pub player: Player,
    pub is_ai: bool,
    pub ai_difficulty: Option<AiDifficulty>,
}

impl PlayerConfig {
    pub fn human(player: Player) -> Self {
        Self {
            player,
            is_ai: false,
            ai_difficulty: None,
        }
    }

    pub fn ai(player: Player, difficulty: AiDifficulty) -> Self {
        Self {
            player,
            is_ai: true,
            ai_difficulty: Some(difficulty),
        }
    }

    pub fn display_name(&self) -> String {
        if self.is_ai {
            format!(
                "{} ({})",
                self.player_name(),
                self.ai_difficulty.unwrap().as_str()
            )
        } else {
            self.player_name()
        }
    }

    fn player_name(&self) -> String {
        match self.player {
            Player::Black => "黒".to_string(),
            Player::White => "白".to_string(),
        }
    }
}
