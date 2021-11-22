"""
Author : Toufik FERHAT, Asma DAGMOUNE
This program is used to split the dataset into training and testing dataset.
"""

# Import needed libraries
import os
import numpy as np
import argparse

 # Split the dataset in training (80%) and testing (20%)
def split_dataset(dataset_path, train_size=0.8):
    # Iterate over the directories in the folder "dataset"
    for dir in os.listdir(dataset_path):
        # For each directory, iterate over the spectogram files
        for file in os.listdir(dataset_path + "/" + dir):
            # Get the absolute path of the file for windows and linux
            file_path = os.path.abspath(dataset_path + "/" + dir + "/" + file)
            # IF windows OS is used
            if os.name == 'nt':
                #Delete the double backslash in the path
                file_path = file_path.replace("\\", "/")
    
            # Check if data/train and test directories exist
            if not os.path.exists("data/train/" + dir):
                # Create the directory if it doesn't exist
                os.makedirs("data/train/" + dir)
            if not os.path.exists("data/test/" + dir):
                # Create the directory if it doesn't exist
                os.makedirs("data/test/" + dir)
            
             # Split the dataset in training and testing
            if np.random.random() < train_size:
                # Move the file to the folder "train"
                os.rename(file_path, os.path.abspath("data/train/" + dir + "/" + file))
            else:
                # Move the file to the folder "test"
                os.rename(file_path, os.path.abspath("data/test/" + dir + "/" + file))


def main():
    # Parse the arguments
    parser = argparse.ArgumentParser(description="Split the dataset into training and testing dataset")
    parser.add_argument("--dataset_path", type=str, default="data/dataset", help="Path to the dataset")
    parser.add_argument("--train_size", type=float, default=0.8, help="Train size")
    args = parser.parse_args()

    # Split the dataset
    split_dataset(args.dataset_path, args.train_size)
    print("The data was well splited")

    
if __name__ == "__main__":
    main()
