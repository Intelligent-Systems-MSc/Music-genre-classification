"""
Authors : Toufik FERHAT, Asma DAGMOUNE
This programm is used to create a class that represents a black box model using tensorflox framework. :
    - The model conssists of  CNN , Fully connected layer and a softmax layer :
    - The CNN is used to extract features from spectrogram images and it contains :
        - Convolutional layer with 32 filters and kernel size of 12x8
        - Max pooling layer with kernel size of 4x1
    - The output of CNN is flattened and fed into a fully connected layer with 200 neurons
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
import argparse

# Defining the black box model
class BlackBoxModel:
    def __init__(self, features):
        self.model = None
        self.model_name = "black_box_model"
        self.model_path = "models/" +self.model_name +"/" + features + "/"+ self.model_name + ".h5"
        self.model_weights_path = "models/" + self.model_name + "/" + features+"/"+self.model_name + "_weights.h5"
        self.model_history_path = "models/" +self.model_name +"/"+ features+"/"+self.model_name + "_history.png"
        self.matrix_path = "models/" +self.model_name +"/"+ features+"/"+self.model_name + "_confusion_matrix.png"
        self.model_history = None
    
    
    def build_model(self):
        """
        This function is used to create the black box model :
        """
        # Defining the input layer
        input_layer = tf.keras.layers.Input(shape=(120, 200, 3))
        
        # Defining the convolutional layer
        conv_layer = tf.keras.layers.Conv2D(32, (12, 8), activation='relu')(input_layer)
        max_pool_layer = tf.keras.layers.MaxPool2D((4, 1))(conv_layer)
        
        # Defining the fully connected layer
        fully_connected_layer = tf.keras.layers.Flatten()(max_pool_layer)
        fully_connected_layer = tf.keras.layers.Dense(200, activation='relu')(fully_connected_layer)
        
        # Defining the output layer
        output_layer = tf.keras.layers.Dense(10, activation='softmax')(fully_connected_layer)
        
        # Compiling the model
        self.model = Model(input_layer, output_layer)
    
    def compile_model(self, optimizer, loss, metrics):
        """
        This function is used to compile the black box model :
        """
        # Compile the model
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        
    def train_model(self, train_generator, validation_generator ,epochs=10):
        """
        This function is used to train the black box model :
        """
        # Training the model
        self.model_history = self.model.fit_generator(train_generator, validation_data = validation_generator,  epochs=epochs)
        
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
        
        return test_loss, test_acc
    
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
        

         # Evaluate the model
        test_loss , test_acc = self.evaluate_model(test_generator)
        
        # Add accuracy and loss to the confusion matrix
        plt.text(0, 0, "Accuracy : " + str(test_acc), horizontalalignment="left", color="white")
        plt.text(0, 1, "Loss : " + str(test_loss), horizontalalignment="left", color="white")
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.savefig(self.matrix_path)
        plt.show()
    
    def summary(self):
        """
        This function is used to print the summary of the black box model :
        """
        self.model.summary()

def main():
    
     # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=str, default="melspectrogram", help="The features to use")
    parser.add_argument("--epochs", type=int, default=50, help="The number of epochs to train the model") 
    args = parser.parse_args()

    
    #Fetch the training dataset and the test dataset
    train_generator = fetch_spectogram_dataset("data/images/"+args.features +"/train")
    validation_generator = fetch_spectogram_dataset("data/images/"+args.features +"/validation")
    test_generator = fetch_spectogram_dataset("data/images/"+args.features +"/test")
   
    
    
    # Create the black box model
    black_box_model = BlackBoxModel(args.features)
    
    # Define the optimizer with a learning rate of 0.05
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)

    # Define the loss function
    loss = tf.keras.losses.CategoricalCrossentropy()

    # Define the metrics accruacy 
    metrics = ['accuracy']
    
    # Build the black box model
    black_box_model.build_model()
    
    # Compile the black box model
    black_box_model.compile_model(optimizer, loss, metrics)
    
    # Train the black box model
    black_box_model.train_model(train_generator, validation_generator, epochs = 50)

    # Save the black box model
    black_box_model.save_model()

    # Evaluate the black box model
    black_box_model.evaluate_model(test_generator)

    # Plot the model history
    black_box_model.plot_model_history()
    
    # Plot the confusion matrix
    black_box_model.plot_confusion_matrix(test_generator)
    
    # Print the summary of the black box model
    black_box_model.summary()

if __name__ == "__main__":
    main()
        

