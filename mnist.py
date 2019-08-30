import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


def createModel(fashionOrDigits):
    if fashionOrDigits.upper() == "FASHION":
        fashion_mnist = tf.keras.datasets.fashion_mnist
        (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    elif fashionOrDigits.upper == "DIGIT":
        digit_mnist = tf.keras.datasets.mnist
        (train_images, train_labels), (test_images, test_labels) = digit_mnist.load_data()
    else:
        print("Invalid function input")
        return
        
    train_images = train_images / 255.0
    test_images = test_images / 255.0
        
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(64, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    
    model.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])
    
    model.fit(train_images, train_labels, epochs=6)

    test_loss, test_acc = model.evaluate(test_images, test_labels)

    print('Test accuracy on ', fashionOrDigits, ' MNIST:', test_acc)

    predictions = model.predict(test_images_fashion)






