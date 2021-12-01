

"""
Author : Asma DAGMOUNE
This program aims to implement the time model for the audio project.
It's architecture is based on the paper "Experimenting with Musically Motivated
Convolutional Neural Networks" and it is as follows:
  - CNN of 32 filters with a kernel size of nx1 
  - Max pooling of size 1xN 
  - Output layer is 10 neurons with a softmax activation function
With M : width of the image and n : 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 
"""

# Import needed libraries
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from fetch_training_dataset import fetch_spectogram_dataset
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
import argparse

# Define the time model
class TimeModel(object):
    def __init__(self):
        self.model = None
        self.model_name = "time_model"
        self.model_path = "models/"+ self.model_name +"/"+ self.model_name + ".h5"
        self.model_weights_path = "models/" + self.model_name + "/"+self.model_name + "_weights.h5"
        self.model_history_path = "models/" + self.model_name +"/"+ self.model_name + "_history.png"
        self.model_history = None
    
    def build_model(self, n = 4):
        """
        This function is used to create the time model :
        """
        M = 40
        # Defining the input layer
        input_layer = tf.keras.layers.Input(shape=(40, 80, 3))
        
        # Defining the convolutional layer
        conv_layer = tf.keras.layers.Conv2D(32, (n, 1), activation='relu')(input_layer)
        max_pool_layer = tf.keras.layers.MaxPool2D((1, M))(conv_layer)
        flatten_layer = tf.keras.layers.Flatten()(max_pool_layer)
        dense_layer = tf.keras.layers.Dense(10, activation='softmax')(flatten_layer)

        output_layer = dense_layer
        
        # Compiling the model
        self.model = Model(input_layer, output_layer)
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
    def train_model(self, train_generator, epochs=10):
        """
        This function is used to train the time model :
        """
        # Training the model
        self.model_history = self.model.fit_generator(train_generator, epochs=epochs)
        
    def save_model(self):
        """
        This function is used to save the time model :
        """
        # Save the model
        self.model.save(self.model_path)
        self.model.save_weights(self.model_weights_path)
        
    def load_model(self):
        """
        This function is used to load the time model :
        """
        # Load the model
        self.model = tf.keras.models.load_model(self.model_path)
        self.model.load_weights(self.model_weights_path)
    
    def save_model_weights(self):
        """
        This function is used to save the time model weights :
        """
        # Save the model weights
        self.model.save_weights(self.model_weights_path)
    
    def evaluate_model(self, test_generator):
        """
        This function is used to evaluate the time model :
        """
        # Evaluate the model
        test_loss, test_acc = self.model.evaluate_generator(test_generator)
        print('Test accuracy:', test_acc)
    
    
    def plot_model_history(self):
        """
        This function is used to plot the model history :
        """
        # Plot the model history
        plt.plot(self.model_history.history['loss'])
        plt.plot(self.model_history.history['accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Loss', 'Accuracy'], loc='upper left')
        plt.savefig(self.model_history_path)
        plt.show()
    
    def predict_classes(self, spectrogram_images):
        """
        This function is used to predict the classes of the spectrogram images :
        """
        # Predict the classes
        predictions = self.model.predict(spectrogram_images)
        return np.argmax(predictions, axis=1)
    
    def plot_confusion_matrix(self, test_generator, classes=None):
        """
        This function is used to plot the confusion matrix :
        """
        # Predict the classes
        predictions = self.predict_classes(test_generator)
        # Plot the confusion matrix
        cm = confusion_matrix(test_generator.classes, predictions)
        if classes is None:
            classes = test_generator.class_indices.keys()
        plt.figure()
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Confusion matrix")
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        fmt = 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.savefig("models/"+ self.model_name +"/"+"confusion_matrix.png")
        plt.show()

def main():
    
    #Fetch the training dataset and the test dataset
    train_generator = fetch_spectogram_dataset("data/images/melspectorgram/train")
    test_generator = fetch_spectogram_dataset("data/images/melspectorgram/test")
    
    # Create the time model
    time_model = TimeModel()
    
    # Build the time model
    time_model.build_model(n=4)
    
    # Train the time model
    time_model.train_model(train_generator, epochs=5)
    
    # Save the time model
    #time_model.save_model()

    # Evaluate the time model
    time_model.evaluate_model(test_generator)
    
    # Plot the time model history
    time_model.plot_model_history()

    # Plot the confusion matrix
    time_model.plot_confusion_matrix(test_generator)

if __name__ == "__main__":
    main()
      