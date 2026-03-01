use std::fmt::Debug;

pub mod othello;

pub use othello::OthelloGame;

pub const BOARD_SIZE: usize = 8;

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Cell {
    Empty,
    Black,
    White,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Player {
    Black,
    White,
}

impl Player {
    pub fn opposite(&self) -> Player {
        match self {
            Player::Black => Player::White,
            Player::White => Player::Black,
        }
    }

    pub fn to_cell(&self) -> Cell {
        match self {
            Player::Black => Cell::Black,
            Player::White => Cell::White,
        }
    }
}

pub trait Game: Send + Debug {
    fn name(&self) -> &'static str;

    fn board(&self) -> &[[Cell; BOARD_SIZE]; BOARD_SIZE];

    fn current_player(&self) -> Player;

    fn is_game_over(&self) -> bool;

    fn black_count(&self) -> usize;

    fn white_count(&self) -> usize;

    fn is_valid_move(&self, row: usize, col: usize) -> bool;

    fn make_move(&mut self, row: usize, col: usize);

    fn has_valid_moves(&self, player: Player) -> bool;

    fn reset(&mut self);

    fn winner(&self) -> Option<Player> {
        if !self.is_game_over() {
            return None;
        }

        let black = self.black_count();
        let white = self.white_count();

        if black > white {
            Some(Player::Black)
        } else if white > black {
            Some(Player::White)
        } else {
            None
        }
    }

    fn status_text(&self) -> String {
        if self.is_game_over() {
            match self.winner() {
                Some(Player::Black) => {
                    format!(
                        "ゲーム終了 - 黒の勝利！ (黒: {}, 白: {})",
                        self.black_count(),
                        self.white_count()
                    )
                }
                Some(Player::White) => {
                    format!(
                        "ゲーム終了 - 白の勝利！ (黒: {}, 白: {})",
                        self.black_count(),
                        self.white_count()
                    )
                }
                None => {
                    format!(
                        "ゲーム終了 - 引き分け！ (黒: {}, 白: {})",
                        self.black_count(),
                        self.white_count()
                    )
                }
            }
        } else {
            let player = match self.current_player() {
                Player::Black => "黒",
                Player::White => "白",
            };
            format!(
                "{}のターン (黒: {}, 白: {})",
                player,
                self.black_count(),
                self.white_count()
            )
        }
    }
}
