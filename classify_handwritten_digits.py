import tensorflow as tf

# Load data
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Preprocess data (reshape arrays as convolutional layers expect an RGB dimension)
train_images = train_images.reshape(60000, 28, 28, 1)
test_images = test_images.reshape(10000, 28, 28, 1)

# Preprocess data (normalise values)
train_images = train_images / 255.0
test_images = test_images / 255.0

# Build model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),  # reshape multidimensional array to be 1D before fully-connected layer
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

# Configure model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')

# Train model
model.fit(train_images, train_labels, epochs=10)

# Evaluate model
model.evaluate(test_images, test_labels)

# Flatten, Dense, Dense = 0.19
# Conv2D, Flatten, Dense, Dense = 0.098
# Conv2D, MaxPooling2D, Flatten, Dense, Dense = 0.28
# Conv2D, MaxPooling2D, Conv2D, MaxPooling2D, Flatten, Dense, Dense = 0.089