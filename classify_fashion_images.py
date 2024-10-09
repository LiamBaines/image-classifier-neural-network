import random

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Load data
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Preprocess data (normalise values)
train_images = train_images / 255.0
test_images = test_images / 255.0

# Build model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

# Configure model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')

# Train model
model.fit(train_images, train_labels, epochs=10)

# Evaluate model
model.evaluate(test_images, test_labels)

# Make a prediction
image_index = random.randint(0, len(test_images) - 1)
predictions = model.predict(test_images)
predicted_class_index = np.argmax(predictions[image_index])
predicted_class = class_names[predicted_class_index]
print(predicted_class)

# Display the image alongside the prediction
plt.figure()
plt.imshow(test_images[image_index], cmap=plt.cm.binary)  # Display the image
plt.title(f"Predicted: {predicted_class}")  # Add predicted class as title
plt.grid(False)  # Optional, removes the grid
plt.show()
