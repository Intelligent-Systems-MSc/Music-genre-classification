# Music-Genre-Recognition

## Introduction
This repository aims to provide an implementation of the [Experimenting with Musically Motivated
Convolutional Neural Networks](https://nubo.ircam.fr/index.php/s/27NkneQw8oBnY8P) problem with some adjustements.

The problem is to classify music genres based on the spectrogram of a music file.

## Dataset
The dataset is the GTZAN music genre collection composed of **1000** music files from **10** genres.

## Pre-processing
The dataset is pre-processed in the following way :
* For each music file, the spectrogram is extracted from the audio file using the [librosa](https://librosa.github.io/librosa/) library.
* The spectrogram is resized to a fixed size using the tensorflow.keras.preprocessing.image.ImageDataGenerator.flow_from_directory.
* Data splits into test and train are performed using the tensorflow.keras.preprocessing.image.ImageDataGenerator().
* Data are saved as TFRecord 

You can download the processed data from:
* Link : https://mega.nz/folder/emhlSaBI
* Key : BGdnOdVRIe10y54EUXztGw

## Model

## Results

## Conclusion




