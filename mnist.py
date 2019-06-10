import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

#fashion_mnist = keras.datasets.fashion_mnist
mnist = tf.keras.datasets.mnist

# For fashion
#(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# For numbers
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# For fashion
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images / 255.0
test_images = test_images / 255.0

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5)

test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)

predictions = model.predict(test_images)

for i in range(10):

    print('Test image ' + str(i + 1) + ' is: ', test_labels[i])

    print('Model predicts that test image ' + str(i + 1) + ' is: ', np.argmax(predictions[i]))

    print('Displaying test image ' + str(i + 1))
    plt.imshow(test_images[i])
    plt.show()

    print()

