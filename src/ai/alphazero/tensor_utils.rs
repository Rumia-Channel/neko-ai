use burn::tensor::{Device, Tensor, TensorData};

use crate::game::{BOARD_SIZE, Cell, Game, Player};

/// Neural network input: 8x8 board with 3 channels (empty, black, white)
/// Returns a tensor of shape [1, 3, 8, 8]
pub fn board_to_tensor(board: &[[Cell; BOARD_SIZE]; BOARD_SIZE], device: &Device) -> Tensor<4> {
    let mut empty = Vec::with_capacity(BOARD_SIZE * BOARD_SIZE);
    let mut black = Vec::with_capacity(BOARD_SIZE * BOARD_SIZE);
    let mut white = Vec::with_capacity(BOARD_SIZE * BOARD_SIZE);

    for row in board {
        for &cell in row {
            match cell {
                Cell::Empty => {
                    empty.push(1.0f32);
                    black.push(0.0f32);
                    white.push(0.0f32);
                }
                Cell::Black => {
                    empty.push(0.0f32);
                    black.push(1.0f32);
                    white.push(0.0f32);
                }
                Cell::White => {
                    empty.push(0.0f32);
                    black.push(0.0f32);
                    white.push(1.0f32);
                }
            }
        }
    }

    let empty_tensor = Tensor::from_data(
        TensorData::new(empty, [1, 1, BOARD_SIZE, BOARD_SIZE]),
        device,
    );
    let black_tensor = Tensor::from_data(
        TensorData::new(black, [1, 1, BOARD_SIZE, BOARD_SIZE]),
        device,
    );
    let white_tensor = Tensor::from_data(
        TensorData::new(white, [1, 1, BOARD_SIZE, BOARD_SIZE]),
        device,
    );

    Tensor::cat(vec![empty_tensor, black_tensor, white_tensor], 1)
}

/// Convert board and current player to neural network input
pub fn game_to_tensor(game: &dyn Game, device: &Device) -> Tensor<4> {
    game_to_tensor_for(game, device)
}

/// Generic version of `game_to_tensor` to avoid trait-object dispatch when possible.
pub fn game_to_tensor_for<G: Game + ?Sized>(game: &G, device: &Device) -> Tensor<4> {
    let board = game.board();
    let current = game.current_player();

    let mut own = Vec::with_capacity(BOARD_SIZE * BOARD_SIZE);
    let mut opponent = Vec::with_capacity(BOARD_SIZE * BOARD_SIZE);
    let mut empty = Vec::with_capacity(BOARD_SIZE * BOARD_SIZE);

    for row in board.iter() {
        for &cell in row.iter() {
            match cell {
                Cell::Empty => {
                    own.push(0.0f32);
                    opponent.push(0.0f32);
                    empty.push(1.0f32);
                }
                Cell::Black => {
                    if current == Player::Black {
                        own.push(1.0f32);
                        opponent.push(0.0f32);
                    } else {
                        own.push(0.0f32);
                        opponent.push(1.0f32);
                    }
                    empty.push(0.0f32);
                }
                Cell::White => {
                    if current == Player::White {
                        own.push(1.0f32);
                        opponent.push(0.0f32);
                    } else {
                        own.push(0.0f32);
                        opponent.push(1.0f32);
                    }
                    empty.push(0.0f32);
                }
            }
        }
    }

    let own_tensor =
        Tensor::from_data(TensorData::new(own, [1, 1, BOARD_SIZE, BOARD_SIZE]), device);
    let opponent_tensor = Tensor::from_data(
        TensorData::new(opponent, [1, 1, BOARD_SIZE, BOARD_SIZE]),
        device,
    );
    let empty_tensor = Tensor::from_data(
        TensorData::new(empty, [1, 1, BOARD_SIZE, BOARD_SIZE]),
        device,
    );

    Tensor::cat(vec![own_tensor, opponent_tensor, empty_tensor], 1)
}
