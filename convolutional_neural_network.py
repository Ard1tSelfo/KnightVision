import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import classification_report
from tensorflow.keras.optimizers import Adam

import load_data as data
#               0    1    2     3    4    5    6    7    8    9    10   11   12
class_names = ['P', 'R', 'KN', 'B', 'Q', 'K', 'X', 'p', 'r', 'kn', 'b', 'q', 'k']

train_images, train_labels, test_images, test_labels = data.load_training_data()


def build_model(input_shape=(60, 60, 1), num_classes=13):
    model = models.Sequential([
        # First Convolutional Block
        layers.Conv2D(60, (3, 3), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Second Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Third Convolutional Block
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.25),

        # Dense Classifier
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    return model


model = build_model(input_shape=(60, 60, 1), num_classes=13)

optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

history = model.fit(train_images, train_labels, epochs=14,
                    validation_data=(test_images, test_labels), verbose=1)

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()


#============================= MODEL EVALUATION ===============================

# Predict the values from the validation dataset
Y_pred = model.predict(test_images)
# Convert predictions classes to one hot vectors
Y_pred_classes = np.argmax(Y_pred, axis=1)

# Compute the confusion matrix
confusion_mtx = confusion_matrix(test_labels, Y_pred_classes)

# Plot the confusion matrix
plt.figure(figsize=(10,8))
sns.heatmap(confusion_mtx, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

print(classification_report(test_labels, Y_pred_classes, target_names=class_names))

plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()

''''# ERROR ANALYSIS
misclassified_idxs = np.where(Y_pred_classes != test_labels)[0]
sample_idxs = np.random.choice(misclassified_idxs, 10, replace=False)

plt.figure(figsize=(15,5))
for i, idx in enumerate(sample_idxs):
    plt.subplot(2,5,i+1)
    plt.imshow(np.squeeze(test_images[idx]), cmap='gray')
    plt.title(f"True: {class_names[test_labels[idx]]}\nPredicted: {class_names[Y_pred_classes[idx]]}")
    plt.axis('off')
plt.tight_layout()
plt.show()'''
#===============================================================================

# Save the weights
model.save('./trained_models/third_model')

