use eframe::egui;

const BOARD_SIZE: usize = 8;
const CELL_SIZE: f32 = 60.0;

#[derive(Clone, Copy, PartialEq, Debug)]
enum Cell {
    Empty,
    Black,
    White,
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Player {
    Black,
    White,
}

impl Player {
    fn opposite(&self) -> Player {
        match self {
            Player::Black => Player::White,
            Player::White => Player::Black,
        }
    }

    fn to_cell(&self) -> Cell {
        match self {
            Player::Black => Cell::Black,
            Player::White => Cell::White,
        }
    }
}

struct OthelloGame {
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
    fn is_valid_move(&self, row: usize, col: usize, player: Player) -> bool {
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
            if self.check_direction(row, col, dr, dc, player) {
                return true;
            }
        }

        false
    }

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

    fn make_move(&mut self, row: usize, col: usize) {
        if !self.is_valid_move(row, col, self.current_player) {
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

    fn has_valid_moves(&self, player: Player) -> bool {
        for row in 0..BOARD_SIZE {
            for col in 0..BOARD_SIZE {
                if self.is_valid_move(row, col, player) {
                    return true;
                }
            }
        }
        false
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

    fn reset(&mut self) {
        *self = Self::default();
    }
}

struct OthelloApp {
    game: OthelloGame,
}

impl Default for OthelloApp {
    fn default() -> Self {
        Self {
            game: OthelloGame::default(),
        }
    }
}

impl eframe::App for OthelloApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("オセロ (Othello)");
            ui.add_space(10.0);

            // Status display
            let status_text = if self.game.game_over {
                let winner = if self.game.black_count > self.game.white_count {
                    "黒の勝利！"
                } else if self.game.white_count > self.game.black_count {
                    "白の勝利！"
                } else {
                    "引き分け！"
                };
                format!(
                    "ゲーム終了 - {} (黒: {}, 白: {})",
                    winner, self.game.black_count, self.game.white_count
                )
            } else {
                let player = match self.game.current_player {
                    Player::Black => "黒",
                    Player::White => "白",
                };
                format!(
                    "{}のターン (黒: {}, 白: {})",
                    player, self.game.black_count, self.game.white_count
                )
            };

            ui.label(egui::RichText::new(status_text).size(20.0));
            ui.add_space(10.0);

            // Game board
            let board_size = CELL_SIZE * BOARD_SIZE as f32;
            let (response, painter) = ui.allocate_painter(
                egui::Vec2::new(board_size, board_size),
                egui::Sense::click(),
            );

            let board_rect = response.rect;

            // Draw board background
            painter.rect_filled(board_rect, 0.0, egui::Color32::from_rgb(0, 100, 0));

            // Draw grid lines
            for i in 0..=BOARD_SIZE {
                let x = board_rect.min.x + i as f32 * CELL_SIZE;
                let y = board_rect.min.y + i as f32 * CELL_SIZE;

                // Vertical lines
                painter.line_segment(
                    [
                        egui::Pos2::new(x, board_rect.min.y),
                        egui::Pos2::new(x, board_rect.max.y),
                    ],
                    egui::Stroke::new(2.0, egui::Color32::BLACK),
                );

                // Horizontal lines
                painter.line_segment(
                    [
                        egui::Pos2::new(board_rect.min.x, y),
                        egui::Pos2::new(board_rect.max.x, y),
                    ],
                    egui::Stroke::new(2.0, egui::Color32::BLACK),
                );
            }

            // Draw pieces
            for row in 0..BOARD_SIZE {
                for col in 0..BOARD_SIZE {
                    let cell_rect = egui::Rect::from_min_size(
                        egui::Pos2::new(
                            board_rect.min.x + col as f32 * CELL_SIZE,
                            board_rect.min.y + row as f32 * CELL_SIZE,
                        ),
                        egui::Vec2::new(CELL_SIZE, CELL_SIZE),
                    );

                    match self.game.board[row][col] {
                        Cell::Black => {
                            painter.circle_filled(
                                cell_rect.center(),
                                CELL_SIZE * 0.4,
                                egui::Color32::BLACK,
                            );
                        }
                        Cell::White => {
                            painter.circle_filled(
                                cell_rect.center(),
                                CELL_SIZE * 0.4,
                                egui::Color32::WHITE,
                            );
                        }
                        Cell::Empty => {}
                    }

                    // Highlight valid moves
                    if !self.game.game_over
                        && self.game.is_valid_move(row, col, self.game.current_player)
                    {
                        painter.circle_stroke(
                            cell_rect.center(),
                            CELL_SIZE * 0.1,
                            egui::Stroke::new(2.0, egui::Color32::YELLOW),
                        );
                    }
                }
            }

            // Handle clicks
            if response.clicked() {
                if let Some(pos) = response.interact_pointer_pos() {
                    let relative_x = pos.x - board_rect.min.x;
                    let relative_y = pos.y - board_rect.min.y;

                    let col = (relative_x / CELL_SIZE) as usize;
                    let row = (relative_y / CELL_SIZE) as usize;

                    if row < BOARD_SIZE && col < BOARD_SIZE {
                        self.game.make_move(row, col);
                    }
                }
            }

            ui.add_space(20.0);

            // Reset button
            if ui
                .button(egui::RichText::new("ゲームをリセット").size(16.0))
                .clicked()
            {
                self.game.reset();
            }
        });
    }
}

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([600.0, 700.0]),
        ..Default::default()
    };

    eframe::run_native(
        "オセロ",
        options,
        Box::new(|_cc| Ok(Box::new(OthelloApp::default()))),
    )
}
