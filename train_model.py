import tensorflow as tf
from tensorflow.keras import layers, models
import os

# Create models folder if not exists
os.makedirs("models", exist_ok=True)

# Load MNIST
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize
train_images = train_images / 255.0
test_images = test_images / 255.0

# Build CNN
model = models.Sequential([
    layers.Input(shape=(28, 28)),
    layers.Reshape((28, 28, 1)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("Training model...")
model.fit(train_images, train_labels, epochs=5)

# Save model
model.save("models/mnist_victim_model.keras")

print("Model saved successfully.")
