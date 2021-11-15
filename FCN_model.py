"""
This programm is used to create a class that represents a fully convolutional network using tensorflow framework. :
    - The model consists of  CNN , 4 Fully connected layers and 4 max pooling :
    - The CNN is used to extract features from spectrogram images and it contains :
        - Convolutional layer with 128 filters and kernel size of 3x3
        - Max pooling layer with kernel size of 2x4       
        - Convolutional layer with 384 filters and kernel size of 3x3
        - Max pooling layer with kernel size of 4x5
        - Convolutional layer with 768 filters and kernel size of 3x3
        - Max pooling layer with kernel size of 3x8
        - Convolutional layer with 2048 filters and kernel size of 3x3
        - Max pooling layer with kernel size of 4x8
    - The output of CNN is flattened and fed into a fully connected layer with 10 neurons with a sigmoide activation function
"""
#import requiring libraries
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from fetch_training_dataset import fetch_spectogram_dataset

#create a class that represents a fully convolutional network
class FCN_model(Model):
    #initialize the class
    def __init__(self, input_shape, num_classes):
        #initialize the model
        super(FCN_model, self).__init__()
        #define the input layer
        self.input_layer = tf.keras.layers.Input(input_shape)
        #define the first convolutional layer
        self.conv1 = Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='relu')(self.input_layer)
        #define the first max pooling layer
        self.maxpool1 = MaxPooling2D(pool_size=(2,4))(self.conv1)
        #define the second convolutional layer
        self.conv2 = Conv2D(filters=384, kernel_size=(3,3), padding='same', activation='relu')(self.maxpool1)
        #define the second max pooling layer
        self.maxpool2 = MaxPooling2D(pool_size=(4,5))(self.conv2)
        #define the third convolutional layer
        self.conv3 = Conv2D(filters=768, kernel_size=(3,3), padding='same', activation='relu')(self.maxpool2)
        #define the third max pooling layer
        self.maxpool3 = MaxPooling2D(pool_size=(3,8))(self.conv3)
        #define the fourth convolutional layer
        self.conv4 = Conv2D(filters=2048, kernel_size=(3,3), padding='same', activation='relu')(self.maxpool3)
        #define the fourth max pooling layer
        self.maxpool4 = MaxPooling2D(pool_size=(4,8))(self.conv4)
        #define the flatten layer
        self.flatten = Flatten()(self.maxpool4)
        #define the first fully connected layer
        self.fc1 = Dense(units=10, activation='sigmoid')(self.flatten)
        #define the output layer
        self.output_layer = Dense(units=num_classes, activation='softmax')(self.fc1)
        #define the model
        self.model = Model(inputs=self.input_layer, outputs=self.output_layer)
        #compile the model
        self.model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
        #define the model checkpoint
        self.model_checkpoint = ModelCheckpoint('FCN_model.h5', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
        #define the model callbacks
        self.model_callbacks = [self.model_checkpoint]
        #define the model data generator
        self.model_data_generator = ImageDataGenerator(rescale=1./255)  # rescale to 0-1
        #define the training data generator
        self.training_data_generator = self.model_data_generator.flow_from_directory(
            directory='../../Data/Training_data',
            target_size=(128,128),
            batch_size=32,
            class_mode='categorical'
        )
        #define the validation data generator
        self.validation_data_generator = self.model_data_generator.flow_from_directory(
            directory='../../Data/Validation_data',
            target_size=(128,128),
            batch_size=32,
            class_mode='categorical'
        )
    #train the model
    def train_model(self, epochs):
        #train the model
        self.model.fit_generator(
            self.training_data_generator,
            steps_per_epoch=100,
            epochs=epochs,
            validation_data=self.validation_data_generator,
            validation_steps=100,
            callbacks=self.model_callbacks
        )
    #save the model
    def save_model(self):
        #save the model
        self.model.save('FCN_model.h5')
    #load the model
    def load_model(self):
        #load the model
        self.model = tf.keras.models.load_model('FCN_model.h5')
        #compile the model




