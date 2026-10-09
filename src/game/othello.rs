use super::{BOARD_SIZE, Cell, Game, Player};

#[derive(Debug, Clone)]
pub struct OthelloGame {
    board: [[Cell; BOARD_SIZE]; BOARD_SIZE],
    current_player: Player,
    game_over: bool,
    black_count: usize,
    white_count: usize,
}

impl Default for OthelloGame {
    fn default() -> Self {
        let mut game = Self {
            board: [[Cell::Empty; BOARD_SIZE]; BOARD_SIZE],
            current_player: Player::Black,
            game_over: false,
            black_count: 2,
            white_count: 2,
        };

        // Initial setup
        game.board[3][3] = Cell::White;
        game.board[3][4] = Cell::Black;
        game.board[4][3] = Cell::Black;
        game.board[4][4] = Cell::White;

        game
    }
}

impl OthelloGame {
    fn check_direction(
        &self,
        row: usize,
        col: usize,
        dr: isize,
        dc: isize,
        player: Player,
    ) -> bool {
        let opponent = player.opposite();
        let mut r = row as isize + dr;
        let mut c = col as isize + dc;
        let mut found_opponent = false;

        while r >= 0 && r < BOARD_SIZE as isize && c >= 0 && c < BOARD_SIZE as isize {
            match self.board[r as usize][c as usize] {
                Cell::Empty => return false,
                cell if cell == opponent.to_cell() => found_opponent = true,
                cell if cell == player.to_cell() => return found_opponent,
                _ => return false,
            }
            r += dr;
            c += dc;
        }

        false
    }

    fn flip_direction(&mut self, row: usize, col: usize, dr: isize, dc: isize, player: Player) {
        if !self.check_direction(row, col, dr, dc, player) {
            return;
        }

        let mut r = row as isize + dr;
        let mut c = col as isize + dc;

        while r >= 0 && r < BOARD_SIZE as isize && c >= 0 && c < BOARD_SIZE as isize {
            if self.board[r as usize][c as usize] == player.to_cell() {
                break;
            }
            self.board[r as usize][c as usize] = player.to_cell();
            r += dr;
            c += dc;
        }
    }

    fn count_pieces(&mut self) {
        self.black_count = 0;
        self.white_count = 0;
        for row in 0..BOARD_SIZE {
            for col in 0..BOARD_SIZE {
                match self.board[row][col] {
                    Cell::Black => self.black_count += 1,
                    Cell::White => self.white_count += 1,
                    _ => {}
                }
            }
        }
    }
}

impl super::CloneGame for OthelloGame {
    fn clone_box(&self) -> Box<dyn Game> {
        Box::new(self.clone())
    }
}

impl Game for OthelloGame {
    fn name(&self) -> &'static str {
        "オセロ"
    }

    fn board(&self) -> &[[Cell; BOARD_SIZE]; BOARD_SIZE] {
        &self.board
    }

    fn current_player(&self) -> Player {
        self.current_player
    }

    fn is_game_over(&self) -> bool {
        self.game_over
    }

    fn black_count(&self) -> usize {
        self.black_count
    }

    fn white_count(&self) -> usize {
        self.white_count
    }

    fn is_valid_move(&self, row: usize, col: usize) -> bool {
        if self.board[row][col] != Cell::Empty {
            return false;
        }

        let directions = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ];

        for &(dr, dc) in &directions {
            if self.check_direction(row, col, dr, dc, self.current_player) {
                return true;
            }
        }

        false
    }

    fn make_move(&mut self, row: usize, col: usize) {
        if !self.is_valid_move(row, col) {
            return;
        }

        let directions = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ];

        self.board[row][col] = self.current_player.to_cell();

        for &(dr, dc) in &directions {
            self.flip_direction(row, col, dr, dc, self.current_player);
        }

        self.current_player = self.current_player.opposite();
        self.count_pieces();

        // Check if current player has any valid moves
        if !self.has_valid_moves(self.current_player) {
            // Switch back if no valid moves
            self.current_player = self.current_player.opposite();
            if !self.has_valid_moves(self.current_player) {
                self.game_over = true;
            }
        }
    }

    fn pass_turn(&mut self) {
        if self.game_over {
            return;
        }

        // Switch to opponent.
        self.current_player = self.current_player.opposite();

        // If opponent also has no valid moves, the game is over.
        if !self.has_valid_moves(self.current_player) {
            self.current_player = self.current_player.opposite();
            if !self.has_valid_moves(self.current_player) {
                self.game_over = true;
            }
        }
    }

    fn has_valid_moves(&self, player: Player) -> bool {
        for row in 0..BOARD_SIZE {
            for col in 0..BOARD_SIZE {
                if self.board[row][col] != Cell::Empty {
                    continue;
                }

                let directions = [
                    (-1, -1),
                    (-1, 0),
                    (-1, 1),
                    (0, -1),
                    (0, 1),
                    (1, -1),
                    (1, 0),
                    (1, 1),
                ];

                for &(dr, dc) in &directions {
                    if self.check_direction(row, col, dr, dc, player) {
                        return true;
                    }
                }
            }
        }
        false
    }

    fn reset(&mut self) {
        *self = Self::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_moves(game: &OthelloGame) -> Vec<(usize, usize)> {
        (0..BOARD_SIZE)
            .flat_map(|row| (0..BOARD_SIZE).map(move |col| (row, col)))
            .filter(|(row, col)| game.is_valid_move(*row, *col))
            .collect()
    }

    #[test]
    fn initial_position_is_standard() {
        let game = OthelloGame::default();

        assert_eq!(game.current_player(), Player::Black);
        assert!(!game.is_game_over());
        assert_eq!(game.black_count(), 2);
        assert_eq!(game.white_count(), 2);
        assert_eq!(game.board()[3][3], Cell::White);
        assert_eq!(game.board()[3][4], Cell::Black);
        assert_eq!(game.board()[4][3], Cell::Black);
        assert_eq!(game.board()[4][4], Cell::White);
    }

    #[test]
    fn black_has_four_opening_moves() {
        let game = OthelloGame::default();
        let mut moves = valid_moves(&game);
        moves.sort_unstable();

        assert_eq!(moves, vec![(2, 3), (3, 2), (4, 5), (5, 4)]);
    }

    #[test]
    fn making_a_move_flips_bracketed_stones() {
        let mut game = OthelloGame::default();
        game.make_move(2, 3);

        // (3,3) の白が挟まれて黒になる
        assert_eq!(game.board()[2][3], Cell::Black);
        assert_eq!(game.board()[3][3], Cell::Black);
        assert_eq!(game.black_count(), 4);
        assert_eq!(game.white_count(), 1);
        assert_eq!(game.current_player(), Player::White);
    }

    #[test]
    fn invalid_move_is_ignored() {
        let mut game = OthelloGame::default();
        let before = *game.board();

        game.make_move(0, 0);

        assert_eq!(*game.board(), before);
        assert_eq!(game.current_player(), Player::Black);
    }

    #[test]
    fn pass_turn_ends_game_when_neither_player_can_move() {
        let mut game = OthelloGame::default();
        for row in game.board.iter_mut() {
            for cell in row.iter_mut() {
                *cell = Cell::Empty;
            }
        }
        // 互いに挟めない位置へ 1 石ずつ置く
        game.board[0][0] = Cell::Black;
        game.board[7][7] = Cell::White;
        game.current_player = Player::Black;
        game.count_pieces();

        assert!(!game.has_valid_moves(Player::Black));
        assert!(!game.has_valid_moves(Player::White));

        game.pass_turn();

        assert!(game.is_game_over());
        assert_eq!(game.black_count(), 1);
        assert_eq!(game.white_count(), 1);
        // 1:1 の引き分け
        assert_eq!(game.winner(), None);
    }

    #[test]
    fn pass_turn_hands_over_the_turn() {
        let mut game = OthelloGame::default();
        // 黒が打てる初期局面でパスすると白の手番になる（ゲームは継続）
        game.pass_turn();

        assert_eq!(game.current_player(), Player::White);
        assert!(!game.is_game_over());
    }

    #[test]
    fn reset_restores_the_initial_position() {
        let mut game = OthelloGame::default();
        game.make_move(2, 3);
        game.make_move(2, 2);

        game.reset();

        assert_eq!(game.black_count(), 2);
        assert_eq!(game.white_count(), 2);
        assert_eq!(game.current_player(), Player::Black);
        assert!(!game.is_game_over());
    }
}
