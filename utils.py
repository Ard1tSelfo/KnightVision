from matplotlib import pyplot as plt
import numpy as np


def plot_chessboard_with_evaluation(chessboard_image, evaluation, top_moves=None):
    square_size = chessboard_image.shape[0] // 8

    # Function to convert algebraic chess notation to pixel coordinates
    def move_to_pixels(move):
        col_map = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7}
        start_col, start_row, end_col, end_row = move[0], int(move[1]) - 1, move[2], int(move[3]) - 1
        start_x = (col_map[start_col] + 0.5) * square_size
        start_y = ((7 - start_row) + 0.5) * square_size
        end_x = (col_map[end_col] + 0.5) * square_size
        end_y = ((7 - end_row) + 0.5) * square_size
        return (start_x, start_y), (end_x, end_y)

    # Create a figure with custom dimensions
    fig = plt.figure(figsize=(8, 6))

    # Add chessboard subplot
    ax_chessboard = fig.add_axes([0.1, 0.1, 0.6, 0.8])  # Adjusted for better layout
    ax_chessboard.imshow(chessboard_image, cmap='gray')
    ax_chessboard.axis('off')  # Turn off axis numbers and ticks

    # Display top moves as text annotations on the chessboard
    if top_moves:
        for move in top_moves:
            start, end = move_to_pixels(move['Move'])
            ax_chessboard.annotate("", xy=end, xycoords='data', xytext=start, textcoords='data',
                                   arrowprops=dict(arrowstyle="->", color="yellow", lw=2))

    # Add evaluation indicator subplot
    ax_evaluation = fig.add_axes([0.75, 0.1, 0.05, 0.8])  # Narrow vertical space for the indicator
    ax_evaluation.plot([0, 0], [-10, 10], color="gray")  # Base line
    ax_evaluation.plot([0], [evaluation], marker="o", markersize=10, color='blue')  # Evaluation "needle"
    ax_evaluation.set_ylim(-10, 10)
    ax_evaluation.set_yticks(np.arange(-10, 11, 2))
    ax_evaluation.set_xticks([])  # Hide the x-axis

    # Adding evaluation scale label
    ax_evaluation.set_ylabel('Evaluation', fontsize=10, labelpad=10)
    ax_evaluation.yaxis.tick_right()  # Move ticks to the right side

    plt.show()


def plot_image(i, predictions_array, img, class_names, coordinate):
    # Note: Removed coordinates from the arguments as we now pass a single coordinate
    img = img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    # Assuming img is 2D (grayscale)
    plt.imshow(img, cmap=plt.cm.gray)

    predicted_label = np.argmax(predictions_array)
    plt.xlabel("{}; {}".format(class_names[predicted_label], coordinate), color='blue')


def plot_results(test_images, predictions, class_names, coordinates):
    # Sort predictions and coordinates based on chessboard layout
    sorted_indices = sorted(range(len(coordinates)), key=lambda i: coordinates[i], reverse=True)
    sorted_predictions = [predictions[i] for i in sorted_indices]
    sorted_coordinates = [coordinates[i] for i in sorted_indices]
    sorted_images = [test_images[i] for i in sorted_indices]

    num_rows = 8
    num_cols = 8
    plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
    for i in range(len(sorted_predictions)):
        # Calculate new subplot index to rotate the grid
        row, col = divmod(i, num_rows)
        row = num_rows - 1 - row
        index = col * num_cols + row + 1
        plt.subplot(num_rows, num_cols, index)
        plot_image(i, sorted_predictions[i], sorted_images, class_names, sorted_coordinates[i])
    plt.tight_layout()
    plt.show()

