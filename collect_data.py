import boardSegmentation
import boardDecomposition

'''IMAGE = "dataset/IMG_3638.jpg"
SEGMENTED_CHESSBOARD = boardSegmentation.segment_chessboard(IMAGE)
boardDecomposition.segment_chessboard(SEGMENTED_CHESSBOARD)

IMAGE = "dataset/IMG_3639.jpg"
SEGMENTED_CHESSBOARD = boardSegmentation.segment_chessboard(IMAGE)
boardDecomposition.segment_chessboard(SEGMENTED_CHESSBOARD)

IMAGE = "dataset/IMG_3640.jpg"
SEGMENTED_CHESSBOARD = boardSegmentation.segment_chessboard(IMAGE)
boardDecomposition.segment_chessboard(SEGMENTED_CHESSBOARD)

IMAGE = "dataset/IMG_3641.jpg"
SEGMENTED_CHESSBOARD = boardSegmentation.segment_chessboard(IMAGE)
boardDecomposition.segment_chessboard(SEGMENTED_CHESSBOARD)

IMAGE = "dataset/IMG_3642.jpg"
SEGMENTED_CHESSBOARD = boardSegmentation.segment_chessboard(IMAGE)
boardDecomposition.segment_chessboard(SEGMENTED_CHESSBOARD)

IMAGE = "dataset/IMG_4076.jpg"
SEGMENTED_CHESSBOARD = boardSegmentation.segment_chessboard(IMAGE)
boardDecomposition.segment_chessboard(SEGMENTED_CHESSBOARD)

IMAGE = "dataset/IMG_4077.jpg"
SEGMENTED_CHESSBOARD = boardSegmentation.segment_chessboard(IMAGE)
boardDecomposition.segment_chessboard(SEGMENTED_CHESSBOARD)

IMAGE = "dataset/IMG_4078.jpg"
SEGMENTED_CHESSBOARD = boardSegmentation.segment_chessboard(IMAGE)
boardDecomposition.segment_chessboard(SEGMENTED_CHESSBOARD)

IMAGE = "dataset/IMG_4079.jpg"
SEGMENTED_CHESSBOARD = boardSegmentation.segment_chessboard(IMAGE)
boardDecomposition.segment_chessboard(SEGMENTED_CHESSBOARD)'''

for i in range(7615, 7663):
    IMAGE = f"dataset/IMG_{i}.jpg"
    SEGMENTED_CHESSBOARD = boardSegmentation.segment_chessboard(IMAGE)
    boardDecomposition.segment_chessboard(SEGMENTED_CHESSBOARD)
    print(IMAGE)
