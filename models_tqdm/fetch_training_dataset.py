"""
Ce programme permet de récuper un jeu de données pour le réseau de neurones à partir de répertoire contenant des spectrogrammes, chaque répertoire représente un genre musical.
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
import time
from tqdm import tqdm
import cv2
import random 
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
# Functions ------------------------------------------------------------------------------------------------------------

def add_image(image_name,data,y,shape_data):

    image = (cv2.imread(image_name, cv2.IMREAD_COLOR)/255)
    image = (cv2.resize(image, shape_data)).astype(np.float16)
    data.append([image,y])


def fetch_spectogram_dataset(path_to_dataset,shape_data):
    """
    This function is used to create spectogram dataset for training and testing :
    """
   
    
    label=0
    data=[]
    for classe in tqdm(os.listdir(path_to_dataset)) :
        print(classe)
        file_name = os.path.join(path_to_dataset,classe)
        
        for name_image in tqdm(os.listdir(file_name)): 
            image= os.path.join(file_name,name_image)
            add_image(image,data,label,shape_data)
        label+=1
    classes = label
    return data, classes


def list_train_test(mat,classes,Nimage,coef):
    Nimage_app = int(Nimage*coef)
    mat_app=[]
    mat_test=[]
    for i in range(classes):
        l=i*Nimage
        mat_app.extend(mat[l:l+Nimage_app])
        mat_test.extend(mat[l+Nimage_app:l+Nimage])
    return  mat_app, mat_test


class prep_data_images: 
    def prep_data(self, path,shape_data):
            data,n_classes = fetch_spectogram_dataset(path,shape_data)
            train,test = list_train_test(data,n_classes,100,0.8)
            random.shuffle(train) # shuffling the training data
            X_train=[]
            y_train=[]

            for features, label in train:
                X_train.append(features)
                y_train.append(label)
                
            width,height,dimension= train[0][0].shape
            X_train=np.array(X_train).reshape(-1,width,height,dimension)
            y_train=to_categorical(y_train,n_classes)

            X_TEST = []
            Y_TEST = []
            for features, label in test:
                X_TEST.append(features)
                Y_TEST.append(label)

            X_TEST=np.array(X_TEST).reshape(-1,width,height,dimension)
            Y_TEST=to_categorical(Y_TEST,n_classes)
            #X_test,X_val,y_test,y_val=train_test_split(X_TEST, Y_TEST, test_size=1/3,random_state=13)
            self.Xtrain = X_train
            self.ytrain = y_train
            self.Xtest = X_TEST
            self.ytest = Y_TEST
# Main --------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    # Fetch spectogram dataset
    data = prep_data_images()
    data.prep_data("images",shape_data)
    print(data.ytest)
    print(data.ytest.shape)
    print(data.ytest.sum())
    



    



