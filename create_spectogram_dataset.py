#Import needed libraries
import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse


"""
Authors: Toufik FERHAT, Asma DAGMOUNE
This script is used to create spectogram dataset for training and testing :
    - The dataset is created from the audio files in the folder "audio_files"
    - The dataset is saved in the folder "dataset"
    - The dataset is used for training and testing the model
"""

def create_spectogram_dataset(audio_files_path, dataset_path):
    # Iterate over the diretories in the folder "genres"
    for dir in tqdm(os.listdir(audio_files_path)):
        # For each directory, iterate over the audio files
        for file in tqdm(os.listdir(audio_files_path+"/" + dir)):
            # Load the audio file
            y, sr = librosa.load(audio_files_path+"/" + dir + "/" + file)
            # Create the spectogram with , nfft = 2048, hop_length = 1024 and Blackman Harris window
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=1024, fmin=0.0, fmax=sr/2, dtype=np.float32)
            
            #Convert the . to - in the file name to avoid problems with the file name
            file_name = file.replace(".", "-")[:-4]
            # Get the absolute path of the file for windows and linux
            file_path = os.path.abspath(dataset_path+"/" + dir)
            # IF windows OS is used
            if os.name == 'nt':
                #Delete the double backslash in the path
                file_path = file_path.replace("\\", "/")
                # Create the directory if it doesn't exist
                if not os.path.exists(file_path):
                    os.makedirs(file_path)
            else:
                # Create the directory if it doesn't exist
                if not os.path.exists(file_path):
                    os.makedirs(file_path)
            # Save the spectogram in the directory "dataset"
            librosa.display.specshow(librosa.power_to_db(S, ref=np.max), fmax=sr/2)
            plt.axis("off")
            name= dataset_path +"/" + dir + "/" + file_name + ".png"
            plt.savefig(name, bbox_inches='tight',pad_inches = 0)
            plt.close()
 
def main():
    # Parse the arguments
    parser = argparse.ArgumentParser(description="Create the spectogram dataset")
    parser.add_argument("--audio_files_path", type=str, default="genres", help="Path to the audio files")
    parser.add_argument("--dataset_path", type=str, default="dataset", help="Path to the dataset")
    args = parser.parse_args()

    # Create the spectogram dataset

    create_spectogram_dataset(args.audio_files_path, args.dataset_path)

if __name__ == "__main__":
    main()
     





