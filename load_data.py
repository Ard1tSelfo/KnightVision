import os
import cv2
import numpy as np
import re
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

data_augmentation = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,  # degrees
    width_shift_range=0.1,  # fraction of total width
    height_shift_range=0.1,  # fraction of total height
    zoom_range=0.1,  # zoom range [1-zoom_range, 1+zoom_range]
    brightness_range=[0.8, 1.2],  # range to choose the brightness factor
    shear_range=5,  # shear intensity (in degrees)
    fill_mode='nearest'  # strategy to fill newly created pixels
)

def load_training_data():

    train_images = []
    train_labels = []
    validation_images = []
    validation_labels = []

    for filename in os.listdir("dataset/data_train/BISHOP"):
        img = cv2.imread(os.path.join("dataset/data_train/BISHOP", filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (60, 60), interpolation=cv2.INTER_AREA)
            train_images.append(img)
            train_labels.append(3)

    for filename in os.listdir("dataset/data_train/bishop_b"):
        img = cv2.imread(os.path.join("dataset/data_train/bishop_b", filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (60, 60), interpolation=cv2.INTER_AREA)
            train_images.append(img)
            train_labels.append(10)

    for filename in os.listdir("dataset/data_train/EMPTY"):
        img = cv2.imread(os.path.join("dataset/data_train/EMPTY", filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (60, 60), interpolation=cv2.INTER_AREA)
            train_images.append(img)
            train_labels.append(6)

    for filename in os.listdir("dataset/data_train/KING"):
        img = cv2.imread(os.path.join("dataset/data_train/KING", filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (60, 60), interpolation=cv2.INTER_AREA)
            train_images.append(img)
            train_labels.append(5)

    for filename in os.listdir("dataset/data_train/king_b"):
        img = cv2.imread(os.path.join("dataset/data_train/king_b", filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (60, 60), interpolation=cv2.INTER_AREA)
            train_images.append(img)
            train_labels.append(12)

    for filename in os.listdir("dataset/data_train/KNIGHT"):
        img = cv2.imread(os.path.join("dataset/data_train/KNIGHT", filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (60, 60), interpolation=cv2.INTER_AREA)
            train_images.append(img)
            train_labels.append(2)

    for filename in os.listdir("dataset/data_train/knight_b"):
        img = cv2.imread(os.path.join("dataset/data_train/knight_b", filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (60, 60), interpolation=cv2.INTER_AREA)
            train_images.append(img)
            train_labels.append(9)

    for filename in os.listdir("dataset/data_train/PAWN"):
        img = cv2.imread(os.path.join("dataset/data_train/PAWN", filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (60, 60), interpolation=cv2.INTER_AREA)
            train_images.append(img)
            train_labels.append(0)

    for filename in os.listdir("dataset/data_train/pawn_b"):
        img = cv2.imread(os.path.join("dataset/data_train/pawn_b", filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (60, 60), interpolation=cv2.INTER_AREA)
            train_images.append(img)
            train_labels.append(7)

    for filename in os.listdir("dataset/data_train/QUEEN"):
        img = cv2.imread(os.path.join("dataset/data_train/QUEEN", filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (60, 60), interpolation=cv2.INTER_AREA)
            train_images.append(img)
            train_labels.append(4)

    for filename in os.listdir("dataset/data_train/queen_b"):
        img = cv2.imread(os.path.join("dataset/data_train/queen_b", filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (60, 60), interpolation=cv2.INTER_AREA)
            train_images.append(img)
            train_labels.append(11)

    for filename in os.listdir("dataset/data_train/ROCK"):
        img = cv2.imread(os.path.join("dataset/data_train/ROCK", filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (60, 60), interpolation=cv2.INTER_AREA)
            train_images.append(img)
            train_labels.append(1)

    for filename in os.listdir("dataset/data_train/rock_b"):
        img = cv2.imread(os.path.join("dataset/data_train/rock_b", filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (60, 60), interpolation=cv2.INTER_AREA)
            train_images.append(img)
            train_labels.append(8)

    # load validation data
    for filename in os.listdir("dataset/data_train/test_images/BISHOP"):
        img = cv2.imread(os.path.join("dataset/data_train/test_images/BISHOP", filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (60, 60), interpolation=cv2.INTER_AREA)
            validation_images.append(img)
            validation_labels.append(3)

    for filename in os.listdir("dataset/data_train/test_images/bishop_b"):
        img = cv2.imread(os.path.join("dataset/data_train/test_images/bishop_b", filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (60, 60), interpolation=cv2.INTER_AREA)
            validation_images.append(img)
            validation_labels.append(10)

    for filename in os.listdir("dataset/data_train/test_images/EMPTY"):
        img = cv2.imread(os.path.join("dataset/data_train/test_images/EMPTY", filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (60, 60), interpolation=cv2.INTER_AREA)
            validation_images.append(img)
            validation_labels.append(6)

    for filename in os.listdir("dataset/data_train/test_images/KING"):
        img = cv2.imread(os.path.join("dataset/data_train/test_images/KING", filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (60, 60), interpolation=cv2.INTER_AREA)
            validation_images.append(img)
            validation_labels.append(5)

    for filename in os.listdir("dataset/data_train/test_images/king_b"):
        img = cv2.imread(os.path.join("dataset/data_train/test_images/king_b", filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (60, 60), interpolation=cv2.INTER_AREA)
            validation_images.append(img)
            validation_labels.append(12)

    for filename in os.listdir("dataset/data_train/test_images/KNIGHT"):
        img = cv2.imread(os.path.join("dataset/data_train/test_images/KNIGHT", filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (60, 60), interpolation=cv2.INTER_AREA)
            validation_images.append(img)
            validation_labels.append(2)

    for filename in os.listdir("dataset/data_train/test_images/knight_b"):
        img = cv2.imread(os.path.join("dataset/data_train/test_images/knight_b", filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (60, 60), interpolation=cv2.INTER_AREA)
            validation_images.append(img)
            validation_labels.append(9)

    for filename in os.listdir("dataset/data_train/test_images/PAWN"):
        img = cv2.imread(os.path.join("dataset/data_train/test_images/PAWN", filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (60, 60), interpolation=cv2.INTER_AREA)
            validation_images.append(img)
            validation_labels.append(0)

    for filename in os.listdir("dataset/data_train/test_images/pawn_b"):
        img = cv2.imread(os.path.join("dataset/data_train/test_images/pawn_b", filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (60, 60), interpolation=cv2.INTER_AREA)
            validation_images.append(img)
            validation_labels.append(7)

    for filename in os.listdir("dataset/data_train/test_images/QUEEN"):
        img = cv2.imread(os.path.join("dataset/data_train/test_images/QUEEN", filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (60, 60), interpolation=cv2.INTER_AREA)
            validation_images.append(img)
            validation_labels.append(4)

    for filename in os.listdir("dataset/data_train/test_images/queen_b"):
        img = cv2.imread(os.path.join("dataset/data_train/test_images/queen_b", filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (60, 60), interpolation=cv2.INTER_AREA)
            validation_images.append(img)
            validation_labels.append(11)

    for filename in os.listdir("dataset/data_train/test_images/ROCK"):
        img = cv2.imread(os.path.join("dataset/data_train/test_images/ROCK", filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (60, 60), interpolation=cv2.INTER_AREA)
            validation_images.append(img)
            validation_labels.append(1)

    for filename in os.listdir("dataset/data_train/test_images/rock_b"):
        img = cv2.imread(os.path.join("dataset/data_train/test_images/rock_b", filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (60, 60), interpolation=cv2.INTER_AREA)
            validation_images.append(img)
            validation_labels.append(8)

    train_images = np.array(train_images)
    train_labels = np.array(train_labels)
    validation_images = np.array(validation_images)
    validation_labels = np.array(validation_labels)

    #train_images = np.expand_dims(train_images, axis=-1)
    #validation_images = np.expand_dims(validation_images, axis=-1)

    # DATA AUGMENTATION
    #train_images = data_augmentation.flow(train_images, train_labels, batch_size=32)

    # Normalize pixel values between 0 and 1
    train_images = train_images / 255.0
    validation_images = validation_images / 255.0

    return train_images, train_labels, validation_images, validation_labels


def extract_image_data():
    validation_images = []
    coordinates = []

    for filename in os.listdir("dataset/data_train/test_images/TEST"):
        img = cv2.imread(os.path.join("dataset/data_train/test_images/TEST", filename), cv2.IMREAD_GRAYSCALE)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        c = re.search(r"\d{1,2}:(.*?)(?=.png)", filename)
        if img is not None:
            intensity = np.mean(img)
            img = img / 255.0
            img = cv2.resize(img, (60, 60), interpolation=cv2.INTER_AREA)
            validation_images.append(img)
            coordinates.append(c.group(1))
    validation_images = np.array(validation_images)
    return validation_images, coordinates
