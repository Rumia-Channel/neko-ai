use crate::game::{Cell, Game, BOARD_SIZE};
use eframe::egui;

pub struct GameApp {
    game: Box<dyn Game>,
}

impl Default for GameApp {
    fn default() -> Self {
        Self {
            game: Box::new(crate::game::OthelloGame::default()),
        }
    }
}

impl GameApp {
    #[allow(dead_code)]
    pub fn new(game: Box<dyn Game>) -> Self {
        Self { game }
    }
}

impl eframe::App for GameApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading(self.game.name());
            ui.add_space(10.0);

            // Status display
            ui.label(egui::RichText::new(self.game.status_text()).size(20.0));
            ui.add_space(10.0);

            // Calculate available space for the board
            let available_size = ui.available_size();
            let min_dimension = available_size.x.min(available_size.y - 60.0); // Reserve space for button
            let board_size = min_dimension.max(100.0); // Minimum board size
            let cell_size = board_size / BOARD_SIZE as f32;

            // Center the board horizontally
            ui.horizontal(|ui| {
                ui.add_space((available_size.x - board_size) / 2.0);

                let (response, painter) = ui.allocate_painter(
                    egui::Vec2::new(board_size, board_size),
                    egui::Sense::click(),
                );

                let board_rect = response.rect;

                // Draw board background
                painter.rect_filled(board_rect, 0.0, egui::Color32::from_rgb(0, 100, 0));

                // Draw grid lines
                for i in 0..=BOARD_SIZE {
                    let x = board_rect.min.x + i as f32 * cell_size;
                    let y = board_rect.min.y + i as f32 * cell_size;

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
                let board = self.game.board();
                for row in 0..BOARD_SIZE {
                    for col in 0..BOARD_SIZE {
                        let cell_rect = egui::Rect::from_min_size(
                            egui::Pos2::new(
                                board_rect.min.x + col as f32 * cell_size,
                                board_rect.min.y + row as f32 * cell_size,
                            ),
                            egui::Vec2::new(cell_size, cell_size),
                        );

                        match board[row][col] {
                            Cell::Black => {
                                painter.circle_filled(
                                    cell_rect.center(),
                                    cell_size * 0.4,
                                    egui::Color32::BLACK,
                                );
                            }
                            Cell::White => {
                                painter.circle_filled(
                                    cell_rect.center(),
                                    cell_size * 0.4,
                                    egui::Color32::WHITE,
                                );
                            }
                            Cell::Empty => {}
                        }

                        // Highlight valid moves
                        if !self.game.is_game_over() && self.game.is_valid_move(row, col) {
                            painter.circle_stroke(
                                cell_rect.center(),
                                cell_size * 0.1,
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

                        let col = (relative_x / cell_size) as usize;
                        let row = (relative_y / cell_size) as usize;

                        if row < BOARD_SIZE && col < BOARD_SIZE {
                            self.game.make_move(row, col);
                        }
                    }
                }
            });

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
