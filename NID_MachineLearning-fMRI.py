#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 08:55:30 2018

@author: Wei Huang, Chong Wang
"""

import numpy as np
from sklearn.ensemble    import RandomForestClassifier
from sklearn.neighbors   import KNeighborsClassifier
from sklearn.ensemble    import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm         import SVC



Classifier = "RandomForest"


"""============================================================================
   The raw data read constains data and lables. """
TRAIN      = np.load('F:/Natural_images_classify/methods/s_zhengzifeng/Train_feature.npy')
TEST       = np.load('F:/Natural_images_classify/methods/s_zhengzifeng/Test_feature.npy')
VALIDATION = np.load('F:/Natural_images_classify/methods/s_zhengzifeng/Validation_feature.npy')

Train_label      = np.load('F:/Natural_images_classify/methods/Train_Label.npy')
Test_Label       = np.load('F:/Natural_images_classify/methods/Test_Label.npy')
Validation_Label = np.load('F:/Natural_images_classify/methods/Validation_Label.npy')

def accuracy_compute(prediction,label):
    count=0
    length=len(prediction)
    for i in range (length):
        if prediction[i]==label[i]:
            count=count+1
    return count/length
        

"""========================================================================="""
if Classifier=="RandomForest":
    clf = RandomForestClassifier()
elif Classifier=="KNeighbors":
    clf = KNeighborsClassifier() 
elif Classifier=="AdaBoost":
    clf = AdaBoostClassifier()
elif Classifier=="bayes":
    clf = GaussianNB() 
elif Classifier=="RBFSVM":
    clf = SVC()
else:
    clf = SVC(kernel="linear") 



clf.fit(TRAIN,Train_label)
Test_prediction = clf.predict(TEST)
accuracy_test = accuracy_compute(Test_prediction,Test_Label)
Validation_prediction = clf.predict(VALIDATION)
accuracy_validation = accuracy_compute(Validation_prediction,Validation_Label)








