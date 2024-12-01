import io
import csv
import zstandard as zstd
import chess.pgn
import chess.engine

engine = chess.engine.SimpleEngine.popen_uci("./stockfish-macos-x86-64-sse41-popcnt")

def games(compressed_games_file):
  decompressor = zstd.ZstdDecompressor()

  with open(compressed_games_file, 'rb') as compressed_games:
    games_binary = decompressor.stream_reader(compressed_games)
    decoded_games = games_binary.read().decode('utf-8')
    games_string_io = io.StringIO(decoded_games)

    while True:
      game = chess.pgn.read_game(games_string_io)

      if game is None:
        break

      yield game

def evaluate(board):
  score = engine.analyse(board, chess.engine.Limit(depth=15))['score'].relative

  if score.is_mate():
    # > 0 if white to mate
    return 10000 if score.mate() > 0 else -10000
  else:
    return score.score()

def eval_change(before_eval, after_eval):
  # handles improvements in black's position (e.g. -34 -> -40)
  if before_eval < 0 and after_eval < 0:
    return abs(after_eval) - abs(before_eval)
  else:
    return after_eval - before_eval

def extract_whites_pawn_moves():
    pawn_moves = []

    for i, game in enumerate(games("lichess_db_standard_rated_2013-01.pgn.zst"), 1):
      print(f"Game {i}")
      board = game.board()
      for move in game.mainline_moves():
        pawn_move = board.piece_at(move.from_square).piece_type == chess.PAWN

        if pawn_move:
          board_state = board.fen()
          before_eval = evaluate(board)

        # we add all moves to the board, not just pawn moves
        board.push(move)

        if pawn_move:
          # after evaluation is from opponents perspective, so we * -1
          after_eval = -(evaluate(board))
          pawn_moves.append({
                    "board_state": board_state,
                    "pawn_move": move.uci(),
                    "before_eval": before_eval,
                    "after_eval": after_eval,
                    "eval_change": eval_change(before_eval, after_eval),
                  })

      if i == 250:
        break

    with open("pawn_moves.csv", "w", newline="") as csv_file:
      writer = csv.DictWriter(
          csv_file,
          fieldnames=["board_state", "pawn_move", "before_eval", "after_eval", "eval_change"]
         )
      writer.writeheader()  # Write column headers
      writer.writerows(pawn_moves)  # Write data rows

extract_whites_pawn_moves()

engine.quit()