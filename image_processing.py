import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import tasks

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


train_images = train_images/255.0
test_images = test_images/255.0

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=10)

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=1)

print('Test accuracy:', test_acc*100,"%")

while True:
    n = int(input("Enter image the image no. for prediction: "))
    predictions = model.predict(test_images)
    plt.figure()
    plt.imshow(test_images[n])
    plt.colorbar()
    plt.show()
    print("The class for the image is: "+class_names[np.argmax(predictions[n])])
    if input("Do you want to continue?(y/n): ") == 'n':
        break


