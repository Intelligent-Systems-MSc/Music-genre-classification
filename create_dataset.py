#Import needed libraries
import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse


"""
Authors: Toufik FERHAT, Asma DAGMOUNE , Rayane KADEM
This script is used to create audio features dataset : 
    - Mel-spectrogram
    - MFCC
    - Chroma-stft
    - Chroma-cqt
    - Chroma-cens

The audio files dataset consists of 10 folders which correspond to 10 different genres and each folder contains 100 different audio files.
According to the choosed feature, it will create a dataset of images representing the audio files.
"""

def create_dataset(dataset_path_in, dataset_path_out, feature, n_fft, hop_length):
    """
    This function is used to create dataset of images.
    :param dataset_path_in: path of the dataset
    :param dataset_path_out : path of image dataset
    :param feature: feature to extract
    :param n_fft: number of fft points
    :param hop_length: number of samples between successive frames
    :return:
    """
    # Create the folders
    if not os.path.exists(dataset_path_out):
        os.makedirs(dataset_path_out)
    if not os.path.exists(dataset_path_out + "/" + feature):
        os.makedirs(dataset_path_out + "/" + feature)

    # Get the list of folders
    folders = os.listdir(dataset_path_in)
    # For each folder
    for folder in tqdm(folders):
        # Get the list of audio files
        files = os.listdir(dataset_path_in + "/" + folder)
        # For each audio file
        for file in tqdm(files):
            # Get the path of the audio file
            path = dataset_path_in + "/" + folder + "/" + file
            # Read the audio file
            y, sr = librosa.load(path)
            # Extract the feature
            if feature == "melspectrogram":
                image = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
            elif feature == "mfcc":
                image = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            elif feature == "chroma_stft":
                image = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
            elif feature == "chroma_cqt":
                image = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
            elif feature == "chroma_cens":
                image = librosa.feature.chroma_cens(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
            
            # Ommit the .wav file extension file
            file = file[:-4]

            # Check if the folder exists
            if not os.path.exists(dataset_path_out + "/" + feature + "/" + folder):
                os.makedirs(dataset_path_out + "/" + feature + "/" + folder)
            
            # Librosa display 
            librosa.display.specshow(librosa.power_to_db(image, ref = np.max), sr =sr , hop_length = hop_length)
            # Ommit axis  
            plt.axis('off')
            plt.axis('tight')
            # Set padding to 0
            plt.tight_layout(pad=0)
                 
            # Save the image
            plt.savefig(dataset_path_out + "/" + feature + "/" + folder + "/" + file + ".png")
            # Close the plot
            plt.close()
            


        
def main():
    
    # Create the parser
    parser = argparse.ArgumentParser(description="Create the dataset of images")
    # Add the arguments
    parser.add_argument("--dataset_path_in", type=str, default="data/genres" ,help="Path of the dataset")
    parser.add_argument("--dataset_path_out", type=str, default="data/dataset" ,help="Path of the dataset")
    parser.add_argument("--feature", type=str,default= "melspectrogram" ,help="Feature to extract : melspectrogram, mfcc, chroma_stft, chroma_cqt, chroma_cens")
    parser.add_argument("--n_fft", type=int, default = 1024,help="Number of fft points")
    parser.add_argument("--hop_length", type=int, default = 512,help="Number of samples between successive frames")

    # Parse the arguments
    args = parser.parse_args()
    # Create the dataset
    create_dataset(args.dataset_path_in, args.dataset_path_out, args.feature, args.n_fft, args.hop_length)


if __name__ == "__main__":
    main()