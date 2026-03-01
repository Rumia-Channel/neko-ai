use super::{BOARD_SIZE, Cell, Game, Player};

#[derive(Debug)]
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
