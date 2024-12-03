import pandas as pd
import numpy as np
import chess
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

df = pd.read_csv('pawn_moves.csv')
scaler = StandardScaler()

def fen_to_matrix(fen):
  num_squares = 64
  matrix = [None] * num_squares
  board = chess.Board(fen)

  for sq in chess.SQUARES:
    piece_value = board.piece_type_at(sq) if board.piece_type_at(sq) else 0
    piece_value = piece_value * (-1 if board.color_at(sq)==False else 1)
    matrix[sq] = round(piece_value / 6, 5)

  return matrix
  # return [matrix[i:i+8] for i in range(0, num_squares, 8)]

def pawn_move_to_list(move):
  move = chess.Move.from_uci(move)
  return [round(move.from_square / 63, 5), round(move.to_square / 63, 5)]

df['normalized_board_matrix'] = df['board_state'].apply(fen_to_matrix)
df['normalized_pawn_move_list'] = df['pawn_move'].apply(pawn_move_to_list)
df['normalized_eval_change'] = scaler.fit_transform(df[['eval_change']])

print(df.head())

X = df[['normalized_board_matrix', 'normalized_eval_change']]
y = df['normalized_pawn_move_list']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_combined = np.hstack((X_train['normalized_board_matrix'].tolist(), X_train['normalized_eval_change'].values.reshape(-1, 1)))

X_test_combined = np.hstack((X_test['normalized_board_matrix'].tolist(), X_test['normalized_eval_change'].values.reshape(-1, 1)))

model = Sequential([
    Dense(128, activation='relu', input_shape=(65,)),  # 64 for board + 1 for eval_change
    Dropout(0.2),  # reduce overfitting
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(2, activation='linear'),
])

# Adam (Adaptive Moment Estimation)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

history = model.fit(X_train_combined, np.array(y_train.tolist()), epochs=40, batch_size=32, validation_split=0.2)

test_loss, test_mae = model.evaluate(X_test_combined, np.array(y_test.tolist()))
print(f"Test Loss: {test_loss}, Test MAE: {test_mae}")

