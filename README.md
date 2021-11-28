# Music-Genre-Recognition

## Introduction
This repository aims to provide an implementation of the [Experimenting with Musically Motivated
Convolutional Neural Networks](https://nubo.ircam.fr/index.php/s/27NkneQw8oBnY8P) problem with some adjustements.

The problem is to classify music genres based on the audio features representation of a music file.

## Dataset
The dataset is the GTZAN music genre collection composed of **1000** music files from **10** genres.


## Pre-processing
The dataset is pre-processed in the following way :
* For each music file, the spectrogram is extracted from the audio file using the [librosa](https://librosa.github.io/librosa/) library.
* The spectrogram is resized to a fixed size using the tensorflow.keras.preprocessing.image.ImageDataGenerator.flow_from_directory.
* Data splits into test and train are performed using spltit_dataset.py script.
### Usage 
 To generate the spectrograms and the data splits, do as follows :
 * Run the "download_extract_GTZAN" file depending on your system to download the dataset and extract it.
 * Run the following command :
```bash
python create_spectrograms.py --audio_files_path=data/genres --dataset_path=data/dataset/ 
```
* Split the dataset into train and test using the `split_dataset.py` script.
```bash
python split_dataset.py --dataset_path=data/dataset/ --train_size=0.8
```

## Models
The models are implemented using the [tensorflow.keras](https://www.tensorflow.org/api_docs/python/tf/keras) library.
We use the folowing models :
### Black Box Model 
The black box model is a convolutional neural network with a fixed architecture inspired by the [SPECTRAL CONVOLUTIONAL NEURAL NETWORK FOR 
MUSIC CLASSIFICATION](https://publik.tuwien.ac.at/files/publik_255986.pdf) architecture and improved by Experimenting with [Musically Motivated Convolutional Neural Networks](http://jordipons.me/media/CBMI16.pdf)

The architecture is defined in the [black_box_mmodel](model_architecture.py) file and it is as follows :
* The input layer is a 2D convolutional layer with 32 filters, a kernel size of 12x8
* A max pooling layer is added after the convolutional layer of size 4x1
* The convolutional layer is flattened and connected to a dense layer with 200 neurons
* Finally, the dense layer is connected to a softmax layer with 10 neurons to output a 10-dimensional vector

### Time model


## Results

## Conclusion




