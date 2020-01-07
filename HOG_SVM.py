import cv2 as cv
import numpy as np
import os
import pandas as pd
import csv

from sklearn.cluster import MiniBatchKMeans
from sklearn.neural_network import MLPClassifier
from skimage import color
from skimage.io import imread, imshow
from skimage.transform import resize
from skimage.feature import hog
from skimage import exposure
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC




#Calculate HOG features for training data
training_X = []
rootdir = r'C:\Users\Ahmed Almostafa\Desktop\Cornell\fall 2019\computer vision\project\places365standard_easyformat\places365_standard\train'
count = 0
invalid_count = 0
invalid_path =[]
filename = r'C:\Users\Ahmed Almostafa\Desktop\Cornell\fall 2019\computer vision\project\HOG_decriptor_365_1200_100_train.txt'
#txt_file =open(filename,'ab')
with open(filename, "a") as txt_file:
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            filepath = subdir + os.sep + file

            try:
                if filepath.endswith(".jpg"):
                    count = count + 1

                    img = imread(filepath)
                    fd = hog(img, orientations=8, pixels_per_cell=(16,16), 
                                                cells_per_block=(16, 16), block_norm= 'L2', multichannel=True)
                    training_X.append(fd)
                    fd = np.array([fd])
                    np.savetxt(txt_file, fd, fmt="%s")
                    txt_file.write(filepath + '\n')

            except:
                print(filepath)
                temp = np.ones(2048)
                temp = -temp
                temp = np.array([temp])
                np.savetxt(txt_file, temp, fmt="%s")
                txt_file.write(filepath + '\n')
                invalid_count = invalid_count + 1
                invalid_path.append(filepath)



            print(count)
            
print('done')            
txt_file.close()   






#Calculate HOG features for testing data
testing_X = []
rootdir = r'C:\Users\Ahmed Almostafa\Desktop\Cornell\fall 2019\computer vision\project\places365standard_easyformat\places365_standard\val'
count = 0
invalid_count = 0
invalid_path =[]
filename = r'C:\Users\Ahmed Almostafa\Desktop\Cornell\fall 2019\computer vision\project\HOG_decriptor_365_1200_100_val.txt'
#txt_file =open(filename,'ab')
with open(filename, "a") as txt_file:
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            filepath = subdir + os.sep + file

            try:
                if filepath.endswith(".jpg"):
                    count = count + 1

                    img = imread(filepath)
                    fd = hog(img, orientations=8, pixels_per_cell=(16,16), 
                                                cells_per_block=(16, 16), block_norm= 'L2', multichannel=True)
                    testing_X.append(fd)
                    fd = np.array([fd])
                    np.savetxt(txt_file, fd, fmt="%s")
                    txt_file.write(filepath + '\n')

            except:
                print(filepath)
                temp = np.ones(2048)
                temp = -temp
                temp = np.array([temp])
                np.savetxt(txt_file, temp, fmt="%s")
                txt_file.write(filepath + '\n')
                invalid_count = invalid_count + 1
                invalid_path.append(filepath)



            print(count)
            
print('done')            
txt_file.close()   




#Create labels for training set
training_Y = np.zeros(438000)
category_idx = 0


for i in range(438000):
    if i%1200 == 0 and i != 0.0:
        category_idx = category_idx + 1
    training_Y[i] = category_idx



#Create labels for testing set
testing_Y = np.zeros(36500)
category_idx = 0


for i in range(36500):
    if i%100 == 0 and i != 0.0:
        category_idx = category_idx + 1
    testing_Y[i] = category_idx


#Fitting SVM
svm_model = OneVsRestClassifier(LinearSVC(random_state=0))
svm_model.fit(training_X, training_Y)        


#Predicting labels for testing data
pred_y = svm_model.predict(testing_X)




#Calculating confusion matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(testing_Y, pred_y)
recall = np.diag(cm) / np.sum(cm, axis = 1)
precision = np.diag(cm) / np.sum(cm, axis = 0)


#Calculate precision and recall
precision = np.diag(cm) / np.sum(cm, axis = 0)
np.nanmean(precision)

recall = np.diag(cm) / np.sum(cm, axis = 1)
np.mean(recall)
