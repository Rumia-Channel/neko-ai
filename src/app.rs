use crate::ai::{AiDifficulty, AiPlayer, GameMode, PlayerConfig, create_ai};
use crate::game::{BOARD_SIZE, Cell, Game, Player};
use eframe::egui;
use std::sync::mpsc::{Receiver, TryRecvError};
use std::time::{Duration, Instant};

const AI_DELAY_MS_AI_VS_AI: u128 = 80;
const AI_DELAY_MS_HUMAN_VS_AI: u128 = 0;

/// 表示用のプレイヤー名
fn player_label(player: Player) -> &'static str {
    match player {
        Player::Black => "黒",
        Player::White => "白",
    }
}

/// AI 思考の結果（プレイヤー, AI本体, 着手）
type AiJobResult = (Player, Box<dyn AiPlayer>, Option<(usize, usize)>);

/// バックグラウンドで進行中の AI 思考
struct AiJob {
    /// 思考しているプレイヤー
    player: Player,
    /// 思考を開始した時刻（経過時間の表示用）
    started_at: Instant,
    /// 思考結果を受け取るチャネル
    receiver: Receiver<AiJobResult>,
}

pub struct GameApp {
    game: Box<dyn Game>,
    game_mode: GameMode,
    ai_difficulty: AiDifficulty,
    /// 手番ごとの AI インスタンス（黒 = 0, 白 = 1）。初回の手番で遅延生成する。
    ai_players: [Option<Box<dyn AiPlayer>>; 2],
    show_settings: bool,
    /// AI がバックグラウンドで思考中なら Some
    thinking: Option<AiJob>,
    last_ai_move_time: Option<Instant>,
}

impl Default for GameApp {
    fn default() -> Self {
        Self {
            game: Box::new(crate::game::OthelloGame::default()),
            game_mode: GameMode::HumanVsHuman,
            ai_difficulty: AiDifficulty::Easy,
            ai_players: [None, None],
            show_settings: true,
            thinking: None,
            last_ai_move_time: None,
        }
    }
}

impl GameApp {
    /// プレイヤーに対応するスロット番号
    fn player_index(player: Player) -> usize {
        match player {
            Player::Black => 0,
            Player::White => 1,
        }
    }

    #[allow(dead_code)]
    pub fn new(game: Box<dyn Game>) -> Self {
        Self {
            game,
            ..Self::default()
        }
    }

    fn reset_game(&mut self) {
        self.game.reset();
        // 進行中の思考結果は破棄する（ワーカースレッドは結果を送れずに終了する）
        self.thinking = None;
        // AI は初回の手番で生成し直す（難易度変更を反映させるため）
        self.setup_ai();
        self.last_ai_move_time = None;
    }

    /// AI インスタンスを破棄する。実際の生成は初回の手番で遅延して行う。
    fn setup_ai(&mut self) {
        self.ai_players = [None, None];
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

    /// バックグラウンドの思考が完了していれば着手を反映する。
    fn poll_ai_job(&mut self) {
        let Some(job) = self.thinking.take() else {
            return;
        };

        match job.receiver.try_recv() {
            Ok((player, ai, choice)) => {
                self.ai_players[Self::player_index(player)] = Some(ai);
                match choice {
                    Some((row, col)) => self.game.make_move(row, col),
                    // 有効手が無い場合はパスしてゲーム終了判定に進む
                    None => self.game.pass_turn(),
                }
                self.last_ai_move_time = Some(Instant::now());
                println!(
                    "AI ({}) 思考時間: {} ms",
                    player_label(player),
                    job.started_at.elapsed().as_millis()
                );
            }
            Err(TryRecvError::Empty) => {
                // まだ思考中
                self.thinking = Some(job);
            }
            Err(TryRecvError::Disconnected) => {
                // ワーカースレッドが落ちた場合は AI を作り直して復帰する
                eprintln!("AI スレッドが結果を返しませんでした。AI を再作成します。");
                self.ai_players[Self::player_index(job.player)] =
                    Some(create_ai(self.ai_difficulty, job.player));
            }
        }
    }

    /// AI の手番ならバックグラウンドで思考を開始する。
    fn maybe_start_ai_job(&mut self, ctx: &egui::Context) {
        if self.show_settings || self.thinking.is_some() || !self.is_ai_turn() {
            return;
        }

        // AI vs AI は目で追えるように着手間隔を空ける
        let delay_ms = if self.game_mode == GameMode::AiVsAi {
            AI_DELAY_MS_AI_VS_AI
        } else {
            AI_DELAY_MS_HUMAN_VS_AI
        };
        if let Some(last) = self.last_ai_move_time {
            let elapsed = last.elapsed().as_millis();
            if elapsed < delay_ms {
                ctx.request_repaint_after(Duration::from_millis((delay_ms - elapsed) as u64));
                return;
            }
        }

        let player = self.game.current_player();
        let index = Self::player_index(player);
        // AI は初回のみ生成する（AlphaZero はモデル読み込みを伴うため毎手作り直さない）
        let ai = match self.ai_players[index].take() {
            Some(ai) => ai,
            None => create_ai(self.ai_difficulty, player),
        };

        let game_snapshot = self.game.clone_box();
        let (sender, receiver) = std::sync::mpsc::channel();
        let repaint_ctx = ctx.clone();

        // 思考は重い（AlphaZero は数秒かかる）ので UI スレッドを止めない
        std::thread::spawn(move || {
            let choice = ai.choose_move(game_snapshot.as_ref());
            let _ = sender.send((player, ai, choice));
            repaint_ctx.request_repaint();
        });

        self.thinking = Some(AiJob {
            player,
            started_at: Instant::now(),
            receiver,
        });
        ctx.request_repaint();
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
            for difficulty in [
                AiDifficulty::Easy,
                AiDifficulty::Medium,
                AiDifficulty::SlightlyHard,
                AiDifficulty::Hard,
                AiDifficulty::Honrou,
            ] {
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
    /// 描画の前に毎フレーム呼ばれる（ウィンドウが隠れている間も呼ばれる）。
    /// AI の思考はここで進める（UI を一切描かないので安全に呼べる）。
    fn logic(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // AI の手番は UI をブロックせずバックグラウンドで進める
        self.poll_ai_job();
        self.maybe_start_ai_job(ctx);

        // 思考中の経過時間表示を更新するため定期的に再描画する
        if self.thinking.is_some() {
            ctx.request_repaint_after(Duration::from_millis(100));
        }
    }

    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        // 渡される Ui は余白も背景も持たないため、従来どおり CentralPanel で包む
        egui::CentralPanel::default().show(ui, |ui| {
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

            // AI の思考中は経過時間を表示する（バックグラウンドで計算中）
            if let Some(job) = &self.thinking {
                ui.label(
                    egui::RichText::new(format!(
                        "AI ({}) 考え中... {:.1}s",
                        player_label(job.player),
                        job.started_at.elapsed().as_secs_f32()
                    ))
                    .italics(),
                );
            }
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

#[cfg(test)]
mod tests {
    use super::*;

    /// テスト用: 設定画面を閉じた状態の GameApp を作る
    fn app_with_mode(game_mode: GameMode, ai_difficulty: AiDifficulty) -> GameApp {
        GameApp {
            game_mode,
            ai_difficulty,
            show_settings: false,
            ..Default::default()
        }
    }

    /// AI の思考がバックグラウンドで進み、選ばれた手が盤面に反映されること。
    #[test]
    fn ai_move_is_applied_from_the_background_thread() {
        let mut app = app_with_mode(GameMode::AiVsAi, AiDifficulty::Easy);

        let ctx = egui::Context::default();
        let stones_before = app.game.black_count() + app.game.white_count();

        app.maybe_start_ai_job(&ctx);
        assert!(app.thinking.is_some(), "AI ジョブが開始されていない");

        // ワーカースレッドの完了を待つ（UI スレッドは止まらない）
        for _ in 0..300 {
            app.poll_ai_job();
            if app.last_ai_move_time.is_some() {
                break;
            }
            std::thread::sleep(Duration::from_millis(50));
        }

        assert!(app.thinking.is_none(), "AI ジョブが完了していない");
        assert!(
            app.last_ai_move_time.is_some(),
            "AI の思考結果が反映されていない"
        );
        // 石を 1 つ置くので総数は 1 増える
        assert_eq!(
            app.game.black_count() + app.game.white_count(),
            stones_before + 1
        );
        // 次の手番のために AI インスタンスが返却されていること
        assert!(app.ai_players[GameApp::player_index(Player::Black)].is_some());
    }

    #[test]
    fn no_ai_job_starts_in_settings_or_on_a_human_turn() {
        let ctx = egui::Context::default();

        // 設定画面では思考を開始しない
        let mut app = GameApp {
            game_mode: GameMode::AiVsAi,
            ..Default::default()
        };
        app.maybe_start_ai_job(&ctx);
        assert!(app.thinking.is_none());

        // 人間 vs AI で黒（人間）の手番なら思考を開始しない
        let mut app = app_with_mode(GameMode::HumanVsAi, AiDifficulty::Easy);
        app.maybe_start_ai_job(&ctx);
        assert!(app.thinking.is_none());
        assert!(!app.is_ai_turn());
    }

    #[test]
    fn reset_discards_a_running_ai_job() {
        let mut app = app_with_mode(GameMode::AiVsAi, AiDifficulty::Easy);

        let ctx = egui::Context::default();
        app.maybe_start_ai_job(&ctx);
        assert!(app.thinking.is_some());

        app.reset_game();

        assert!(app.thinking.is_none());
        assert_eq!(app.game.black_count(), 2);
        assert_eq!(app.game.white_count(), 2);
    }
}
