import utils
import tensorflow as tf
import boardSegmentation
import boardDecomposition
import load_data
import numpy as np
import FENgenerator
from chessEngine import ChessEngine

class_names = ['P', 'R', 'N', 'B', 'Q', 'K', '1', 'p', 'r', 'n', 'b', 'q', 'k']
full_class_names = ['W_PAWN', 'W_ROCK', 'W_KNIGHT', 'W_BISHOP', 'W_QUEEN', 'W_KING', 'X', 'B_PAWN', 'B_ROCK',
                    'B_KNIGHT', 'B_BISHOP', 'B_QUEEN', 'B_KING']

IMAGE = "dataset/FIRST_TEST.jpg"
SEGMENTED_CHESSBOARD = boardSegmentation.segment_chessboard(IMAGE)
boardDecomposition.segment_chessboard(SEGMENTED_CHESSBOARD)
test_images, coordinates = load_data.extract_image_data()

# Load trained model instance
model = tf.keras.models.load_model('trained_models/third_model')
# add probability interpretation layer
model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
# feed the image to make predictions
predictions = model.predict(test_images)
predicted_symbols = [class_names[pred] for pred in np.argmax(predictions, axis=1)]
# plot the results
utils.plot_results(test_images, predictions, full_class_names, coordinates)

FENNotation = FENgenerator.build_fen(predicted_symbols, coordinates, True)

engine = ChessEngine(depth=15, elo=2800)
evaluation = engine.get_evaluation(FENNotation)
top_moves = engine.get_top_moves(FENNotation, num_moves=3)

print(FENNotation)
print(evaluation)
print(top_moves)

utils.plot_chessboard_with_evaluation(SEGMENTED_CHESSBOARD, evaluation, top_moves)
