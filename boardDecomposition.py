import cv2
from math import ceil


def segment_chessboard(image):
    coordinates = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    for c in range(0, img.shape[0], img.shape[0]//8):
        for r in range(0, img.shape[1], img.shape[1]//8):
            if r//62 != 8 and c//62 != 8:
                segment_chessboard.counter += 1
                # uncomment to add test/validation images on runtime
                cv2.imwrite(f"dataset/data_train/test_images/TEST/img:{segment_chessboard.counter}:{coordinates[c//62]}{9-((r//62)+1)}.png",
                            img[r:r + ceil(img.shape[0]/8), c:c + ceil(img.shape[1]/8)])

                # uncomment to feed more data in dataset
                # cv2.imwrite(f"dataset/data_train/img:{segment_chessboard.counter}:{coordinates[c//62]}{9-((r//62)+1)}.png",
                #            img[r:r + ceil(img.shape[0]/8), c:c + ceil(img.shape[1]/8)])


segment_chessboard.counter = 0
