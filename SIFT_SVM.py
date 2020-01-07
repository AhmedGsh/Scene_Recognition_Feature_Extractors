import cv2 as cv
import numpy as np
import os
import pandas as pd
import csv
import os
import timeit
from sklearn.cluster import MiniBatchKMeans
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
import numpy as np
import pylab as pl
import pandas as pd
import matplotlib.pyplot as plt 
%matplotlib inline
import seaborn as sns
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.model_selection import cross_val_score, GridSearchCV






#Calculate SIFT and store as txt file
start = timeit.default_timer()
invalid_count = 0
dico = []
rootdir = r'C:\Users\duke\Desktop\CS6670_Final_Project\places365standard_easyformat_SIFT_100\places365_standard\train'
count = 0
filename = r'C:\Users\duke\Desktop\CS6670_Final_Project\SIFT_decriptor_365_1200.txt'
with open(filename, "a") as txt_file:
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            filepath = subdir + os.sep + file
            if filepath.endswith(".jpg"):
                count = count + 1
                try:
                    img = cv.imread(filepath)
                    gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
                    sift = cv.xfeatures2d.SIFT_create()
                    kp, des = sift.detectAndCompute(gray,None)   
                    for d in des:
                        dico.append(d)
                    np.savetxt(txt_file, des, fmt="%s")
                    txt_file.write(filepath + '\n')

                except:
                    invalid_count = invalid_count + 1
                    print(filepath)

                print(count)
            
txt_file.close()   
stop = timeit.default_timer()
sift_cal_time = stop - start
print('done')       
print(sift_cal_time)    


#Create the Visual Bag of Words using MiniBatchKmeans
k = 2048

start = timeit.default_timer()
kmeans = MiniBatchKMeans(n_clusters=k, batch_size=128, verbose=1).fit(dico)
stop = timeit.default_timer()
sift_cal_time = stop - start







#Create feature vector by assinging SIFT descriptors to the 2048 clusters for training set
kmeans.verbose = False

histo_list = []


rootdir = r'C:\Users\Ahmed Almostafa\Desktop\Cornell\fall 2019\computer vision\project\places365standard_easyformat\places365_standard\train'
count = 0
filename = r'C:\Users\Ahmed Almostafa\Desktop\Cornell\fall 2019\computer vision\project\SIFT_decriptor_365_1200_result.txt'
invalid_count_1 = 0
count_1 = 0
    
with open(filename, "a") as txt_file:
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            filepath = subdir + os.sep + file
            if filepath.endswith(".jpg"):
                count_1 = count_1 + 1
                try:
                    img = cv.imread(filepath)
                    gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
                    sift = cv.xfeatures2d.SIFT_create()
                    kp, des = sift.detectAndCompute(gray,None)  
                    histo = np.zeros(k)
                    nkp = np.size(kp)
                    for d in des:
                        idx = kmeans.predict([d])
                        histo[idx] += 1/nkp
                    histo_list.append(histo)
                    histo = np.array([histo])
                    np.savetxt(txt_file, histo, fmt="%s")
                    txt_file.write(filepath + '\n')

                except:
                    invalid_count_1 = invalid_count_1 + 1
                    print(filepath)

                print(count_1)
                
txt_file.close() 



#Do the same thing for testing set
rootdir = r'C:\Users\Ahmed Almostafa\Desktop\Cornell\fall 2019\computer vision\project\places365standard_easyformat\places365_standard\val'
count = 0
filename = r'C:\Users\Ahmed Almostafa\Desktop\Cornell\fall 2019\computer vision\project\SIFT_decriptor_365_1200_100_val.txt'
invalid_count_1 = 0
count_1 = 0
histo_list_test = []
    
with open(filename, "a") as txt_file:
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            filepath = subdir + os.sep + file


            if filepath.endswith(".jpg"):
                count_1 = count_1 + 1
                try:
                    img = cv.imread(filepath)
                    gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
                    sift = cv.xfeatures2d.SIFT_create()
                    kp, des = sift.detectAndCompute(gray,None)  
                    histo = np.zeros(k)
                    nkp = np.size(kp)
                    for d in des:
                        idx = kmeans.predict([d])
                        histo[idx] += 1/nkp
                    histo_list_test.append(histo)
                    histo = np.array([histo])
                    np.savetxt(txt_file, histo, fmt="%s")
                    txt_file.write(filepath + '\n')

                except:
                    invalid_count_1 = invalid_count_1 + 1
                    print(filepath)

                print(count_1)
                
txt_file.close() 


#Create labels for training set
histo_list_Y = np.zeros(438000)
category_idx = 0


for i in range(438000):
    if i%1200 == 0 and i != 0.0:
        category_idx = category_idx + 1
    histo_list_Y[i] = category_idx



#Create labels for testing set
testing_Y = np.zeros(36500)
category_idx = 0


for i in range(36500):
    if i%100 == 0 and i != 0.0:
        category_idx = category_idx + 1
    testing_Y[i] = category_idx
        


#Settings up hyperparameter space and using grid search 
params_grid = [{'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

#Fitting SVM model with cross validation
svm_model = GridSearchCV(SVC(), params_grid, cv=5)
svm_model.fit(histo_list, histo_list_Y)


#Predicting labels for testing dataset
pred_y = svm_model.predict(histo_list_test)



#Calculating confusion matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(histo_list_test, pred_y)
recall = np.diag(cm) / np.sum(cm, axis = 1)
precision = np.diag(cm) / np.sum(cm, axis = 0)


#Calculate precision and recall
precision = np.diag(cm) / np.sum(cm, axis = 0)
np.nanmean(precision)

recall = np.diag(cm) / np.sum(cm, axis = 1)
np.mean(recall)






