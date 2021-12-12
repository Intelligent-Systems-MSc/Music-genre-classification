"""
Authors : Asma DAGMOUNE, FERHAT Toufik
This program allows to retrieve a data set for the neural network from directories containing spectrograms, each directory representing a musical genre.
"""

# Imports --------------------------------------------------------------------------------------------------------------
import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Functions ------------------------------------------------------------------------------------------------------------
def fetch_spectogram_dataset(path_to_dataset):
    """
    This function is used to create spectogram dataset for training and testing :
    """
    # Create ImageDataGenerator object 
    train_datagen = ImageDataGenerator(rescale=1./255)
    # CREATE DATASET
    train_generator = train_datagen.flow_from_directory(path_to_dataset, target_size=(120, 200), batch_size=32, class_mode='categorical', shuffle=False)
    return train_generator


# Main --------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    # Fetch spectogram dataset
    train_generator = fetch_spectogram_dataset()

    



