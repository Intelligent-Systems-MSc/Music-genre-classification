"""
Authors : Rayane KADEM
This programm is used to create a class that represents a black box model using tensorflox framework. :
    - The model conssists of  CNN layer, Fully connected layers and a softmax layer :
    - The CNN is used to extract features from spectrogram images and it contains :
        - 
        -

    - The output of CNN is flattened and fed into a fully connected layer with .. neurons
    - The output of fully connected layer is fed into a softmax layer with 10 neurons
"""

# Importing the required libraries
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from fetch_training_dataset import fetch_spectogram_dataset
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
#import argparse
import keras
from keras.layers import Flatten, Dense, Dropout, Conv2D, Activation ,MaxPooling2D, Flatten
from tensorflow.keras.models import Sequential

# Defining the black box model
class CNNModel:
    def __init__(self):
        self.model = None
        self.model_name = "2layers_CNN_model"
        self.model_path = "models/" + self.model_name +"/"+ self.model_name + ".h5"
        self.model_weights_path = "models/" + self.model_name +"/"+  self.model_name + "_weights.h5"
        self.model_history_path_acc = "models/" + self.model_name +"/"+ self.model_name + "_history_acc.png"
        self.model_history_path_loss = "models/" + self.model_name +"/"+ self.model_name + "_history_loss.png"
        
        self.model_history = None
    
    def build_model(self):
        """
        This function is used to create the black box model :
        """
        # Defining the input layer
        input_layer = tf.keras.layers.Input(shape=(40, 80, 3))
        
        model = Sequential()

        #Adding the CNN layers along with some drop outs and maxpooling
        model.add(Conv2D(64, (3,3), activation = 'relu', input_shape = (40, 80, 3)))
        model.add(MaxPooling2D(pool_size = (2,2)))

        model.add(Conv2D(32, (3,2), activation = 'relu'))
        model.add(MaxPooling2D(pool_size = 2))
        model.add(Flatten())

        
        #Adding the dense layers
        model.add(Dense(256, activation = 'relu'))
        model.add(Dropout(0.5))
        model.add(Dense(124, activation = 'relu'))

        #final output layer with 10 predictions to be made
        model.add(Dense(10, activation = 'softmax'))
    
        
        
        # Compiling the model
        self.model = model
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
        
    def train_model(self, train_generator,validation_generator, epochs=10):
        """
        This function is used to train the black box model :
        """
        # Training the model
        self.model_history = self.model.fit_generator(train_generator, epochs=epochs,validation_data=validation_generator, shuffle=True)

        
    def save_model(self):
        """
        This function is used to save the black box model :
        """
        # Save the model
        self.model.save(self.model_path)
        
    def load_model(self):
        """
        This function is used to load the black box model :
        """
        # Load the model
        self.model = tf.keras.models.load_model(self.model_path)
        
    def save_model_weights(self):
        """
        This function is used to save the weights of the black box model :
        """
        # Save the weights
        self.model.save_weights(self.model_weights_path)
    

    def evaluate_model(self, test_generator):
        """
        This function is used to evaluate the black box model :
        """
        # Evaluate the model
        test_loss, test_acc = self.model.evaluate_generator(test_generator)
        print("Test loss: ", test_loss)
        print("Test accuracy: ", test_acc)
    
    def plot_model_history(self):
        """
        This function is used to plot the model history :
        """
        #define variables
        loss = self.model_history.history['loss']
        val_loss = self.model_history.history['val_loss']
        accuracy = self.model_history.history['accuracy']
        val_accuracy = self.model_history.history['val_accuracy']
        epochs = range(1, len(loss) + 1)
        # Plot the model history
        plt.figure()
        plt.plot(epochs, accuracy, label='Training accuracy')
        plt.plot(epochs, val_accuracy, label='Validation accuracy')
        plt.title('Training and validation accuracy')
        plt.legend()
        plt.savefig(self.model_history_path_acc)
        plt.show()

        plt.figure()
        plt.plot(epochs, loss, label='Training loss')
        plt.plot(epochs, val_loss, label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        plt.savefig(self.model_history_path_loss)
        plt.show()
        
    def predict_class(self, spectrogram_image):
        """
        This function is used to predict the class of the spectrogram image :
        """
        # Predict the class
        prediction = self.model.predict(spectrogram_image)
        return np.argmax(prediction)

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
    
    # Fetch the training and test dataset
    train_generator = fetch_spectogram_dataset("data/train")
    test_generator =  fetch_spectogram_dataset("data/test")
    validation_generator =  fetch_spectogram_dataset("data/validation")
    # Create the black box model
    black_box_model = CNNModel()
    
    # Build the black box model
    black_box_model.build_model()
    
    # Train the black box model
    black_box_model.train_model(train_generator,validation_generator, epochs=100)

    # Save the black box model
    black_box_model.save_model()

    # Evaluate the black box model
    black_box_model.evaluate_model(test_generator)

    # Plot the model history
    black_box_model.plot_model_history()
    
    # Plot the confusion matrix
    black_box_model.plot_confusion_matrix(test_generator)

if __name__ == "__main__":
    main()
        

