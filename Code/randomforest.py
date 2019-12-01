#!/usr/bin/env python
# coding: utf-8

# In[62]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import cv2
import pickle
from sklearn.ensemble import RandomForestClassifier
from skimage.feature import hog
import skimage
import random
import os


# In[57]:


def make_inputs():
    X= []
    Y= []
    for fold in os.listdir('./templates/ranks/'):
        for img in os.listdir(os.path.join('./templates/ranks/', fold)):
            imarray= np.array(cv2.imread('./templates/ranks/' + fold + '/' + img, 0))
            fd, hog_image = hog(imarray, orientations=9, pixels_per_cell=(8, 8), 
                    cells_per_block=(2, 2), visualize=True, multichannel=False)
            hog_image= np.reshape(hog_image, (1, hog_image.shape[0]*hog_image.shape[1]))[0]
            print (hog_image.shape)
            if(hog_image.shape[0]== 700):
#             toput[:len(imarray)]= imarray
                X.append(hog_image)
                Y.append(fold)
#     for fold in os.listdir('./templates/suits/'):
#         for img in os.listdir(os.path.join('./templates/suits/', fold)):
#             imarray= np.array(cv2.imread('./templates/suits/' + fold + '/' + img, 0))
#             print (imarray.shape)
#             imarray= np.reshape(imarray, (1, imarray.shape[0]*imarray.shape[1]))[0]
#             toput= np.zeros(1000)
#             toput[:len(imarray)]= imarray
#             X.append(toput)
#             Y.append(fold)
    
    print (len(X), len(Y))
    return (np.array(X), np.array(Y))


# In[58]:


def make_inputs2():
    X= []
    Y= []
    for fold in os.listdir('./templates/suits/'):
        for img in os.listdir(os.path.join('./templates/suits/', fold)):
            imarray= np.array(cv2.imread('./templates/suits/' + fold + '/' + img, 0))
            fd, hog_image = hog(imarray, orientations=9, pixels_per_cell=(8, 8), 
                    cells_per_block=(2, 2), visualize=True, multichannel=False)
            hog_image= np.reshape(hog_image, (1, hog_image.shape[0]*hog_image.shape[1]))[0]
            print (hog_image.shape)
            if(hog_image.shape[0]== 700):
#             toput= np.zeros(1000)
#             toput[:len(imarray)]= imarray
                X.append(hog_image)
                Y.append(fold)
    
    print (len(X), len(Y))
    return (np.array(X), np.array(Y))


# In[59]:


Xtrain, Ytrain= make_inputs()
Xtrain2, Ytrain2= make_inputs2()


# In[63]:


clf= RandomForestClassifier(n_estimators=500, max_depth=3, random_state=0)
clf.fit(Xtrain, Ytrain)
qclf= RandomForestClassifier(n_estimators=500, max_depth=3, random_state=0)
qclf.fit(Xtrain2,Ytrain2)


# In[64]:


f= open('./model.pkl', 'wb')
qf= open('./model2.pkl', 'wb')
pickle.dump(clf, f)
pickle.dump(qclf, qf)
f.close()
qf.close()


# In[ ]:




