#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import cv2
import skimage
import random
import os


# In[4]:


def rotateImage(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


# In[23]:


def aug(folder):
    img= folder + '1.png'
    imarray= np.array(cv2.imread(img, 0))
    print (imarray.shape)
    degrees= np.arange(-20, 20, 2)[:17]
    for i in range(1, 18):
        fname= folder + str(i+1) + '.png'
        imt= rotateImage(imarray, degrees[i-1])
        cv2.imwrite(fname, imt)
#     cv2.imwrite(folder + str(29) + '.png', np.fliplr(imarray))
#     cv2.imwrite(folder + str(30) + '.png', np.flipud(imarray))


# In[36]:


aug('./templates/ranks/8/')


# In[ ]:




