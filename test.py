# Tensorflow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

# Dataset
#fashion_mnist = keras.datasets.fashion_mnist
mnist = tf.keras.datasets.mnist

# For fashion
#(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# For numbers
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# For fashion
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Normalize
train_images = train_images / 255.0
test_images = test_images / 255.0

# Define structure of network
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

# Define optimizer, loss, & metric
model.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])

# Train model
model.fit(train_images, train_labels, epochs=5)

# Test model
test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)

# Store predictions from test data based on model
predictions = model.predict(test_images)

print('First test image is: ', test_labels[0])

print('Model predicts that first test image is: ', np.argmax(predictions[0]))

