"""
This programm is used to create a class that represents a black box model using tensorflox framework. :
    - The model conssists of  CNN , Fully connected layer and a softmax layer :
    - The CNN is used to extract features from spectrogram images and it contains :
        - Convolutional layer with 32 filters and kernel size of 12x8
        - Max pooling layer with kernel size of 4x1
    - The output of CNN is flattened and fed into a fully connected layer with 200 neurons
    - The output of fully connected layer is fed into a softmax layer with 8 neurons
"""

# Importing the required libraries
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from fetch_training_dataset import fetch_spectogram_dataset


# Creating a class that represents a black box model
class BlackBoxModel(Model):
    """
    This class is used to create a black box model using tensorflow framework.
    """

    def __init__(self):
        """
        This function is used to create a black box model.
        """
        super(BlackBoxModel, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(32, (12, 8), activation='relu')
        self.pool1 = tf.keras.layers.MaxPool2D((4, 1))
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(200, activation='relu')
        self.dense2 = tf.keras.layers.Dense(8, activation='softmax')

    def call(self, x):
        """
        This function is used to call the model.
        :param x: Input data
        :return: Output of the model
        """
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x

    def predict(self, x):
        """
        This function is used to predict the output of the model.
        :param x: Input data
        :return: Output of the model
        """
        return self(x)
    
    def loss(self, x, y):
        """
        This function is used to calculate the loss of the model.
        :param x: Input data
        :param y: Output data
        :return: Loss of the model
        """
        return tf.keras.losses.categorical_crossentropy(y, self(x))
    
    def accuracy(self, x, y):
        """
        This function is used to calculate the accuracy of the model.
        :param x: Input data
        :param y: Output data
        :return: Accuracy of the model
        """
        return tf.keras.metrics.categorical_accuracy(y, self(x))
    
    def train(self, train_generator, epochs=10, batch_size=32, verbose=True):
        """
        This function is used to train the model.
        :param train_generator: Training data
        :param epochs: Number of epochs
        :param batch_size: Batch size
        :param verbose: Verbose
        :return: None
        """
        self.compile(optimizer='adam', loss=self.loss, metrics=[self.accuracy])
        self.fit_generator(train_generator, epochs=epochs, steps_per_epoch=len(train_generator),
                           verbose=verbose, use_multiprocessing=True, workers=4)
    
    def evaluate(self, test_generator, batch_size=32):
        """
        This function is used to evaluate the model.
        :param test_generator: Test data
        :param batch_size: Batch size
        :return: None
        """
        return self.evaluate_generator(test_generator, steps=len(test_generator),
                                       use_multiprocessing=True, workers=4, verbose=1)


# Main --------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    
    # Load the training data
    train_generator = fetch_spectogram_dataset('data/dataset')

    # Instantiate the model
    black_box_model = BlackBoxModel()

    # Train the model
    black_box_model.train(train_generator, epochs=10, batch_size=32, verbose=True)

    # Evaluate the model
    loss, accuracy = black_box_model.evaluate(train_generator, batch_size=32)

    # Print the accuracy
    print('Accuracy: ', accuracy)

    # Save the model
    #black_box_model.save('models/black_box_model.h5')


