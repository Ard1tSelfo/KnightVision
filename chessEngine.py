from stockfish import Stockfish

class ChessEngine:
    def __init__(self, path='/usr/local/bin/stockfish', depth=15, elo=2800):
        # Initialize the Stockfish engine with the specified path, depth, and Elo rating
        self.stockfish = Stockfish(path=path, depth=depth, parameters={'UCI_Elo': elo})

    def get_evaluation(self, FEN):
        # Set the FEN position and return the evaluation value divided by 100.0
        self.stockfish.set_fen_position(FEN)
        return self.stockfish.get_evaluation()['value'] / 100.0

    def get_top_moves(self, FEN, num_moves=3):
        # Set the FEN position and return the top 'num_moves' moves
        self.stockfish.set_fen_position(FEN)
        return self.stockfish.get_top_moves(num_moves)
