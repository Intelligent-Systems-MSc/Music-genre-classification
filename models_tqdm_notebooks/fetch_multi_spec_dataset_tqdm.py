#!/usr/bin/env python
# coding: utf-8
 
"""
Authors : Rayane KADEM
This code allows a multi fetch of spectrogram datasets using tqdm library.
This code takes as an input the names of 3 spectrograms to consider when creating 
the dataset to be used in the multi-spectrogram-representation notebook. for an audio track
it takes 3 selected spectrograms representations images on a gray level, and combines them
into one 3 dimension image to be fed to a CNN.
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

def add_image(image_name1,image_name2,image_name3,data,y,shape_data):

    image1 = (cv2.imread(image_name1, cv2.IMREAD_GRAYSCALE)/255)
    width,height = image1.shape
    image1 = image1.reshape(width,height,1)
    image2 = (cv2.imread(image_name2, cv2.IMREAD_GRAYSCALE)/255).reshape(width,height,1)
    image3 = (cv2.imread(image_name3, cv2.IMREAD_GRAYSCALE)/255).reshape(width,height,1)
    
    #create one image from 3 spectrograms
    image_mid = np.concatenate((image1,image2),axis=2)
    image = np.concatenate((image_mid,image3),axis=2)
    
    if shape_data == None : 
        image = image.astype(np.float16)
    else : 
        image = (cv2.resize(image, shape_data)).astype(np.float16)
    data.append([image,y])


def fetch_spectogram_dataset(path_to_dataset,shape_data,features1,features2,features3):
    """
    This function is used to create spectogram dataset for training and testing :
    """
   
    
    label=0
    data=[]
    #set directory of the 3 types of spectrograms
    file_features1= path_to_dataset + "/"+ features1
    file_features2= path_to_dataset + "/"+ features2
    file_features3= path_to_dataset + "/"+ features3
    
    if os.name == 'nt':
                #Delete the double backslash in the path
                file_features1 = file_features1.replace("/", "\\")
                file_features2 = file_features2.replace("/", "\\")
                file_features3 = file_features3.replace("/", "\\")
    for classe in tqdm(os.listdir(file_features1)) :
        print(classe)
        file_name1 = os.path.join(file_features1,classe)
        file_name2 = os.path.join(file_features2,classe)
        file_name3 = os.path.join(file_features3,classe)
        
        for name_image in tqdm(os.listdir(file_name1)): 
            image1= os.path.join(file_name1,name_image)
            image2= os.path.join(file_name2,name_image)
            image3= os.path.join(file_name3,name_image)
            add_image(image1,image2,image3,data,label,shape_data)
            
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
    def prep_data(self, path,shape_data,features1,features2,features3):
            data,n_classes = fetch_spectogram_dataset(path,shape_data,features1,features2,features3)
            train,test = list_train_test(data,n_classes,100,0.7)
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
            X_test,X_val,y_test,y_val=train_test_split(X_TEST, Y_TEST, test_size=1/3,random_state=13)
            self.Xtrain = X_train
            self.ytrain = y_train
            self.Xtest = X_test
            self.ytest = y_test
            self.Xval = X_val
            self.yval = y_val
# Main --------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    # Fetch spectogram dataset
    path = "data/images"
    shape_data = None
    data,n_classes = fetch_spectogram_dataset(path,shape_data,"cqt","mfcc","melspectrogram")

    




    



