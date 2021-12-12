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
python split_dataset.py
```

To train a model, test it and evaluate it : 
```bash
python <Name of the model>.py --features=<Feauture used> --epochs=<Number of epochs>
```

Note : results of the training and the evaluation are saved in the "models/Name of the model/Feature Used" folder. In each folder, we have the following files :
* Confusion matrix : the confusion matrix of the model.
* Model History : the history of the model.
* Model : the saved weights of the model.



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

The time model is a convolutional neural network with a fixed architecture inspired by the [Musically Motivated Convolutional Neural Networks](http://jordipons.me/media/CBMI16.pdf). It has been concived to be able to classify music genres based on the time domain features. 

The architecture is defined as follows :

* The input layer is a 2D convolutional layer with 32 filters, a kernel size of 1xn
* A max pooling layer is added after the convolutional layer of size Mx1
* The convolutional layer is flattened and connected to a dense layer with 10 neurons with a sigmoid activation function which corresponds to the probability of the music genre

M is the height of the spectrogram and n is a variable to chose.


### Frequency model

The frequency model is the exact versa of the time model but with a kernel size of mx1 instead of 1xn and max pooling layer of size 1xN. 

N is the width of the spectrogram and m is a variable to chose.


### Time-Frequency model

The time-frequency model tries to combine both the time and the frequency models. It is a convolutional neural network with a fixed architecture inspired by the [Musically Motivated Convolutional Neural Networks](http://jordipons.me/media/CBMI16.pdf). 

We use the time and frequency models to extract the features from the spectrogram separately and then we combine to be input of a feed forward neural network of 200 neurons.


### VGG16 transfer Learning Model

The model is built on the ground of the well known VGG16 model, pre-trained on ImageNet dataset. We set the last 8 layers of VGG16 model to be re-trained, and add 3 fully connected Layers to enlarge the classification Block.

### VGG16 + SVM Model 
We take the previous VGG16 transfer learning architecture as feature extractor, and replace the last Sofrmax output layer with an SVM Classifier.

### 2Layers CNN Model 
The model contains 2 CNN Layers, a flatten layer and 2 Fully connected layers before the output softmax Layer.
It resembles to the Black box model but considers deeper convolutions in time and frequencies, with a more sophisticated classification block.

### 3Layers CNN Model 
Very similar to the 2Layers CNN Model. It only adds an extra convolutional layer to deepen the extracted features.


## Notebooks demonstration

In the folder "models_tqdm_notebooks", You find for every visual representation (spectrogram), a notebook that allows to test the different approaches proposed, mainly:  VGG16 transfer learning,
VGG16+SVM, 2layers and 3 Layers CNN architectures. Moreover, for Mel-spectrogram,
you find tested time, frequency, time/frequency models. 

All models in this folder use datasets generated with tqdm library and not ImageGenerator. 
The difference between the two is negligeable, however the structure of the code may differ in 
certain parts.

Due to the high numbers of carried out tests and the very limited storage space on git, drive links of used datasets and results are available in the file "models_tqdm_notebooks/readme.md".