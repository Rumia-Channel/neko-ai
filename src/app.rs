use crate::ai::{AiDifficulty, AiPlayer, GameMode, PlayerConfig, create_ai};
use crate::game::{BOARD_SIZE, Cell, Game, Player};
use eframe::egui;

pub struct GameApp {
    game: Box<dyn Game>,
    game_mode: GameMode,
    ai_difficulty: AiDifficulty,
    ai_player: Option<Box<dyn AiPlayer>>,
    show_settings: bool,
    thinking: bool,
    last_ai_move_time: Option<std::time::Instant>,
}

impl Default for GameApp {
    fn default() -> Self {
        Self {
            game: Box::new(crate::game::OthelloGame::default()),
            game_mode: GameMode::HumanVsHuman,
            ai_difficulty: AiDifficulty::Easy,
            ai_player: None,
            show_settings: true,
            thinking: false,
            last_ai_move_time: None,
        }
    }
}

impl GameApp {
    #[allow(dead_code)]
    pub fn new(game: Box<dyn Game>) -> Self {
        Self {
            game,
            game_mode: GameMode::HumanVsHuman,
            ai_difficulty: AiDifficulty::Easy,
            ai_player: None,
            show_settings: true,
            thinking: false,
            last_ai_move_time: None,
        }
    }

    fn reset_game(&mut self) {
        self.game.reset();
        self.setup_ai();
        self.thinking = false;
        self.last_ai_move_time = None;
    }

    fn setup_ai(&mut self) {
        match self.game_mode {
            GameMode::HumanVsHuman => {
                self.ai_player = None;
            }
            GameMode::HumanVsAi => {
                // AI plays as White (second player)
                self.ai_player = Some(create_ai(self.ai_difficulty, Player::White));
            }
            GameMode::AiVsAi => {
                // Both players are AI - for now, same difficulty
                self.ai_player = Some(create_ai(self.ai_difficulty, self.game.current_player()));
            }
        }
    }

    fn is_ai_turn(&self) -> bool {
        if self.game.is_game_over() {
            return false;
        }

        match self.game_mode {
            GameMode::HumanVsHuman => false,
            GameMode::HumanVsAi => {
                // AI is always White in HumanVsAi mode
                self.game.current_player() == Player::White
            }
            GameMode::AiVsAi => true, // Always AI's turn in AI vs AI
        }
    }

    fn make_ai_move(&mut self) {
        if let (Some(ai), Some((row, col))) = (
            self.ai_player.as_ref(),
            self.ai_player
                .as_ref()
                .unwrap()
                .choose_move(self.game.as_ref()),
        ) {
            let _ = ai;
            self.game.make_move(row, col);
        }
    }

    fn current_player_config(&self) -> PlayerConfig {
        let current = self.game.current_player();

        match self.game_mode {
            GameMode::HumanVsHuman => PlayerConfig::human(current),
            GameMode::HumanVsAi => {
                if current == Player::White {
                    PlayerConfig::ai(current, self.ai_difficulty)
                } else {
                    PlayerConfig::human(current)
                }
            }
            GameMode::AiVsAi => PlayerConfig::ai(current, self.ai_difficulty),
        }
    }

    fn render_settings(&mut self, ui: &mut egui::Ui) {
        ui.heading("ゲーム設定");
        ui.add_space(10.0);

        // Game mode selection
        ui.label("ゲームモード:");
        for mode in [
            GameMode::HumanVsHuman,
            GameMode::HumanVsAi,
            GameMode::AiVsAi,
        ] {
            ui.radio_value(&mut self.game_mode, mode, mode.as_str());
        }
        ui.add_space(10.0);

        // AI difficulty selection (only show if AI is involved)
        if self.game_mode != GameMode::HumanVsHuman {
            ui.label("AIの難易度:");
            for difficulty in [AiDifficulty::Easy, AiDifficulty::Medium, AiDifficulty::Hard] {
                ui.radio_value(&mut self.ai_difficulty, difficulty, difficulty.as_str());
            }
            ui.add_space(10.0);
        }

        // Start game button
        if ui.button("ゲーム開始").clicked() {
            self.show_settings = false;
            self.reset_game();
        }

        ui.add_space(20.0);
        ui.separator();
        ui.add_space(20.0);
    }
}

impl eframe::App for GameApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Handle AI turns
        if self.is_ai_turn() && !self.thinking {
            // Add a small delay for AI moves so it's visible
            let should_move = match self.last_ai_move_time {
                None => true,
                Some(last_time) => last_time.elapsed().as_millis() > 500,
            };

            if should_move {
                self.thinking = true;
                self.make_ai_move();
                self.thinking = false;
                self.last_ai_move_time = Some(std::time::Instant::now());

                // Update AI player reference for AI vs AI mode
                if self.game_mode == GameMode::AiVsAi && !self.game.is_game_over() {
                    self.ai_player =
                        Some(create_ai(self.ai_difficulty, self.game.current_player()));
                }
            }
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            // Show settings panel before game starts
            if self.show_settings {
                self.render_settings(ui);
            }

            ui.heading(self.game.name());
            ui.add_space(10.0);

            // Game mode indicator
            ui.label(format!("モード: {}", self.game_mode.as_str()));

            // Current player indicator
            let player_config = self.current_player_config();
            let player_text = if self.game.is_game_over() {
                "ゲーム終了".to_string()
            } else {
                format!("現在のプレイヤー: {}", player_config.display_name())
            };
            ui.label(egui::RichText::new(player_text).size(18.0));

            // Status display
            ui.label(egui::RichText::new(self.game.status_text()).size(20.0));
            ui.add_space(10.0);

            // Calculate available space for the board
            let available_size = ui.available_size();
            let min_dimension = available_size.x.min(available_size.y - 100.0); // Reserve space for buttons
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
                for (row, row_cells) in board.iter().enumerate().take(BOARD_SIZE) {
                    for (col, cell) in row_cells.iter().enumerate().take(BOARD_SIZE) {
                        let cell_rect = egui::Rect::from_min_size(
                            egui::Pos2::new(
                                board_rect.min.x + col as f32 * cell_size,
                                board_rect.min.y + row as f32 * cell_size,
                            ),
                            egui::Vec2::new(cell_size, cell_size),
                        );

                        match *cell {
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

                        // Highlight valid moves (only for human players)
                        if !self.game.is_game_over()
                            && !self.is_ai_turn()
                            && self.game.is_valid_move(row, col)
                        {
                            painter.circle_stroke(
                                cell_rect.center(),
                                cell_size * 0.1,
                                egui::Stroke::new(2.0, egui::Color32::YELLOW),
                            );
                        }
                    }
                }

                // Handle clicks (only for human players)
                if response.clicked()
                    && !self.is_ai_turn()
                    && let Some(pos) = response.interact_pointer_pos()
                {
                    let relative_x = pos.x - board_rect.min.x;
                    let relative_y = pos.y - board_rect.min.y;

                    let col = (relative_x / cell_size) as usize;
                    let row = (relative_y / cell_size) as usize;

                    if row < BOARD_SIZE && col < BOARD_SIZE {
                        self.game.make_move(row, col);
                    }
                }
            });

            ui.add_space(20.0);

            // Control buttons
            ui.horizontal(|ui| {
                // Reset button
                if ui
                    .button(egui::RichText::new("ゲームをリセット").size(16.0))
                    .clicked()
                {
                    self.reset_game();
                }

                // Settings button
                if ui
                    .button(egui::RichText::new("設定に戻る").size(16.0))
                    .clicked()
                {
                    self.show_settings = true;
                    self.reset_game();
                }
            });
        });
    }
}
