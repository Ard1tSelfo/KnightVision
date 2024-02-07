import cv2
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

model = tf.keras.models.load_model('trained_models/third_model')

layer_names = ['conv2d', 'max_pooling2d', 'conv2d_1', 'conv2d_2', 'max_pooling2d_1', 'conv2d_3', 'conv2d_4']

layer_outputs = [model.get_layer(name).output for name in layer_names]
activation_model = Model(inputs=model.input, outputs=layer_outputs)

# Load the image
img = cv2.imread('dataset/data_train/test_images/KNIGHT/img:359:E2:KNIGHT.png', cv2.IMREAD_GRAYSCALE )
img = cv2.resize(img, (60, 60), interpolation=cv2.INTER_AREA)
img = np.array(img)
img = img / 255.0
# Reshape the image to add the batch and channel dimensions
img = np.expand_dims(img, axis=0)  # Add the batch dimension
img = np.expand_dims(img, axis=-1)  # Add the channel dimension

activations = activation_model.predict(img)

for layer_number, layer_activation in enumerate(activations):  # Iterate over each layer
    n_features = layer_activation.shape[-1]  # Number of features in the feature map
    size = layer_activation.shape[1]  # The feature map has shape (1, size, size, n_features).

    n_cols = n_features // 16  # We will display 16 features per row
    n_cols = max(n_cols, 1)  # Ensure there is at least 1 column
    n_rows = n_features // n_cols  # Calculate the number of rows needed

    display_grid = np.zeros((size * n_rows, size * n_cols))

    for col in range(n_cols):  # Tiles each filter into a big horizontal grid
        for row in range(n_rows):
            channel_image = layer_activation[0, :, :, col * n_rows + row]
            # Post-process the feature to make it visually palatable
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[row * size: (row + 1) * size, col * size: (col + 1) * size] = channel_image

    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))
    plt.title(f'Layer {layer_number + 1} : {layer_names[layer_number]} activations')
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')

plt.show()

