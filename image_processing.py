# Importing necessary libraries
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import tasks

# Loading the Fashion MNIST dataset from Keras
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Class names for the labels in the dataset
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Normalizing the pixel values of images to range [0, 1]
train_images = train_images / 255.0
test_images = test_images / 255.0

# Creating a three-layer neural network using Keras Sequential API
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),  # Flatten the 2D image to a 1D array
    keras.layers.Dense(128, activation='relu'),    # Fully connected layer with 128 units and ReLU activation
    keras.layers.Dense(10, activation='softmax')   # Fully connected layer with 10 units (one for each class) and softmax activation
])

# Compiling the model with the Adam optimizer, sparse categorical crossentropy loss, and accuracy metric
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Training the model with the training data for 10 epochs
model.fit(train_images, train_labels, epochs=10)

# Evaluating the model's performance on the test data
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=1)

# Printing the test accuracy
print('Test accuracy:', test_acc * 100, "%")

# Inferring the class of selected images based on user input
while True:
    n = int(input("Enter the image number for prediction: "))
    predictions = model.predict(test_images)
    
    # Displaying the selected image
    plt.figure()
    plt.imshow(test_images[n])
    plt.colorbar()
    plt.show()
    
    # Printing the predicted class for the selected image
    print("The class for the image is: " + class_names[np.argmax(predictions[n])])
    
    # Asking the user if they want to continue predicting more images
    if input("Do you want to continue? (y/n): ") == 'n':
        break
