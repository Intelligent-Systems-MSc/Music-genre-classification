"""
Author : Toufik FERHAT, Asma DAGMOUNE

The original dataset is formed of 10 folders which correspends to 10 classes, each folder contains 100 audio files
This program is used to split the dataset into training and testing dataset.
"""

# Import needed libraries
import os
import numpy as np
import argparse


def split_dataset(dataset_path_in, path_out, train_size, validation_size):
    """
    Split the dataset into training , validation and testing set
    """
    #  Get the list of all the folders
    folders = os.listdir(dataset_path_in)
    
    # Create the output folders
    if not os.path.exists(path_out):
        os.makedirs(path_out)
    if not os.path.exists(path_out + "/train"):
        os.makedirs(path_out + "/train")
    if not os.path.exists(path_out + "/validation"):
        os.makedirs(path_out + "/validation")
    if not os.path.exists(path_out + "/test"):
        os.makedirs(path_out + "/test")

    for folder in folders:
        # Get the list of all the files in the folder
        files = os.listdir(dataset_path_in + "/" + folder)
        # Get the number of files in the folder
        nb_files = len(files)
        # Get the number of files in the training set
        nb_train_files = int(nb_files * train_size)
        # Get the number of files in the validation set
        nb_validation_files = int(nb_files * validation_size)
        # Get the number of files in the testing set
        nb_test_files = nb_files - nb_train_files - nb_validation_files
        
        # Create the training set
        for i in range(nb_train_files):
            # Check if path exists
            if not os.path.exists(path_out + "/train/" + folder):
                os.makedirs(path_out + "/train/" + folder)
            os.rename(dataset_path_in + "/" + folder + "/" + files[i], path_out + "/train/" + folder + "/" + files[i])
        # Create the validation set
        for i in range(nb_train_files, nb_train_files + nb_validation_files):
            # Check if path exists
            if not os.path.exists(path_out + "/validation/" + folder):
                os.makedirs(path_out + "/validation/" + folder)
            os.rename(dataset_path_in + "/" + folder + "/" + files[i], path_out + "/validation/" + folder + "/" + files[i])
        # Create the testing set
        for i in range(nb_train_files + nb_validation_files, nb_files):
            # Check if path exists
            if not os.path.exists(path_out + "/test/" + folder):
                os.makedirs(path_out + "/test/" + folder)
            os.rename(dataset_path_in + "/" + folder + "/" + files[i], path_out + "/test/" + folder + "/" + files[i])
        

    
    
def main():
    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="data/dataset/melspectrogram/", help="Path to the dataset")
    parser.add_argument("--path_out", type=str, default="data/images/melspectorgram/", help="Path to the output folder")
    parser.add_argument("--train_size", type=float, default=0.8, help="Size of the training set")
    parser.add_argument("--validation_size", type=float, default=0, help="Size of the validation set")
    args = parser.parse_args()

    # Split the dataset
    try :
        split_dataset(args.dataset_path, args.path_out,args.train_size, args.validation_size)
    except Exception as e:
        print(e)
    
    print("Dataset splitted")

    
if __name__ == "__main__":
    main()