import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

class Model:
    def __init__(self, fashionOrDigits):
        pass
    
    def getFashionData(self):
        fashion_mnist = keras.datasets.fashion_mnist
        return fashion_mnist.load_data()
    
    def getDigitData(self):
        digit_mnist = tf.keras.datasets.mnist
        return digit_mnist.load_data()


(train_images_digits, train_labels_digits), (test_images_digits, test_labels_digits) = mnist.load_data()

class_names_fashion = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images_fashion = train_images_fashion / 255.0
test_images_fashion = test_images_fashion / 255.0
train_images_digits = train_images_digits / 255.0
test_images_digits = test_images_digits / 255.0

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])

model.fit(train_images_fashion, train_labels_fashion, epochs=6)

test_loss, test_acc = model.evaluate(test_images_fashion, test_labels_fashion)

print('Test accuracy on Fashion MNIST:', test_acc)

predictions = model.predict(test_images_fashion)




