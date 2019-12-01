#!/usr/bin/env python
# coding: utf-8

# In[18]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import cv2
import pickle
from skimage.feature import hog
import skimage
import random
import os
from scipy import ndimage
import imutils
import math
get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:


def getAngle(a, b, c):
    ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
    return ang + 360 if ang < 0 else ang


# In[7]:


def find_parent(parent, i):
    if(parent[i]==  i):
        return i
    return find_parent(parent, parent[i])


# In[8]:


def shrink_array(cntsarray):
    finalcnts= []
    sumarray= (np.sum(np.sum(np.sum(cntsarray, axis= 1), axis= 1), axis= 1))
    print (sumarray)
    parent= np.arange(len(sumarray))
    print (parent)
    for i in range(len(sumarray)):
        for j in range(len(sumarray)):
            if(i!= j and abs(sumarray[i]-sumarray[j])<= 10):
                a= find_parent(parent, i)
                b= find_parent(parent, j)
                parent[b]= a
    for i in range(len(parent)):
        if(parent[i]== i):
            finalcnts.append(cntsarray[i])
    return finalcnts


# In[9]:


def flattener(image, pts, w, h):
    temp_rect = np.zeros((4,2), dtype = "float32")
    
    s = np.sum(pts, axis = 2)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]

    diff = np.diff(pts, axis = -1)
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    # Need to create an array listing points in order of
    # [top left, top right, bottom right, bottom left]
    # before doing the perspective transform

    if w <= 0.8*h: # If card is vertically oriented
        temp_rect[0] = tl
        temp_rect[1] = tr
        temp_rect[2] = br
        temp_rect[3] = bl

    if w >= 1.2*h: # If card is horizontally oriented
        temp_rect[0] = bl
        temp_rect[1] = tl
        temp_rect[2] = tr
        temp_rect[3] = br

    # If the card is 'diamond' oriented, a different algorithm
    # has to be used to identify which point is top left, top right
    # bottom left, and bottom right.
    
    if w > 0.8*h and w < 1.2*h: #If card is diamond oriented
        # If furthest left point is higher than furthest right point,
        # card is tilted to the left.
        if pts[1][0][1] <= pts[3][0][1]:
            # If card is titled to the left, approxPolyDP returns points
            # in this order: top right, top left, bottom left, bottom right
            temp_rect[0] = pts[1][0] # Top left
            temp_rect[1] = pts[0][0] # Top right
            temp_rect[2] = pts[3][0] # Bottom right
            temp_rect[3] = pts[2][0] # Bottom left

        # If furthest left point is lower than furthest right point,
        # card is tilted to the right
        if pts[1][0][1] > pts[3][0][1]:
            # If card is titled to the right, approxPolyDP returns points
            # in this order: top left, bottom left, bottom right, top right
            temp_rect[0] = pts[0][0] # Top left
            temp_rect[1] = pts[3][0] # Top right
            temp_rect[2] = pts[2][0] # Bottom right
            temp_rect[3] = pts[1][0] # Bottom left
            
        
    maxWidth = 200
    maxHeight = 300
    dst = np.array([[0,0],[maxWidth-1,0],[maxWidth-1,maxHeight-1],[0, maxHeight-1]], np.float32)
    M = cv2.getPerspectiveTransform(temp_rect,dst)
    warp = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warp


# In[10]:


def get_rank(imarray, cnt):
    card= cnt
    x,y,w,h= cv2.boundingRect(card)
    warp1= flattener(imarray, card, w, h)[6:45, 5:34]
#     print (warp1.shape)
    plt.imshow(warp1)
    plt.show()
    return warp1


# In[11]:


def get_suit(imarray, cnt):
    card= cnt
    x,y,w,h= cv2.boundingRect(card)
    warp1= flattener(imarray, card, w, h)[44:77, 5:34]
#     print (warp1.shape)
    plt.imshow(warp1)
    plt.show()
    return warp1


# In[12]:


def card_detect(loc):
    imarray= np.array(cv2.imread(loc, 0))
    print (imarray.shape)
    imblur1= cv2.GaussianBlur(imarray, (5,5), 0)
    imblur2= cv2.GaussianBlur(imarray, (7,7), 0)
    plt.imshow(imblur1, cmap= 'gray')
    plt.title("After Gaussian Blur")
    plt.show()
    
    high_thresh, thresh_im = cv2.threshold(imblur1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    low_thresh = 0.5*high_thresh
    edgearray1= cv2.Canny(imblur1, low_thresh, high_thresh)
    high_thresh, thresh_im = cv2.threshold(imblur2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    low_thresh = 0.5*high_thresh
    edgearray2= cv2.Canny(imblur2, low_thresh, high_thresh)
    plt.imshow(edgearray1, cmap= 'gray')
    plt.title("After Edge detection")
    plt.show()
    cnts1= cv2.findContours(edgearray1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts1= imutils.grab_contours(cnts1)
    cnts2= cv2.findContours(edgearray2.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts2= imutils.grab_contours(cnts2)
    cnts1.extend(cnts2)
    cnts= sorted(cnts1, key = cv2.contourArea, reverse = True)[:15]
    print (len(cnts))
    
    cntsarray= []
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.009*peri, True)
        if (len(approx)== 4):#card will be a rectangle always
            cntsarray.append(approx)
            
    imarraycolor= np.array(cv2.imread(loc))
    print (len(cntsarray))
    cv2.drawContours(imarraycolor, cntsarray, -1, (0, 255, 0), 3)
    plt.imshow(imarraycolor)
    plt.title("Contours detected")
    plt.show()
    cv2.imwrite('./op.png', imarraycolor)
    sumarray= (np.sum(np.sum(np.sum(cntsarray, axis= 1), axis= 1), axis= 1))
    a= shrink_array(cntsarray)
#     print (len(a))
    return a


# In[13]:


def rankplussuit_detect(imarray, cntsarray):
    f= open('./model.pkl', 'rb')
    clf= pickle.load(f)
    qclf=pickle.load(open('./model2.pkl', 'rb'))
    for cnt in cntsarray:
        obsrank= get_rank(imarray, cnt)
        obssuit= get_suit(imarray, cnt)
        rankPath= './templates1/ranks/'
        suitPath= './templates1/suits/'
        obsrankres= cv2.resize(obsrank, (20, 35))
        fd, hog_image = hog(obsrankres, orientations=9, pixels_per_cell=(8, 8), 
                        cells_per_block=(2, 2), visualize=True, multichannel=False)
        hog_image= np.reshape(hog_image, (1, hog_image.shape[0]*hog_image.shape[1]))[0]
        print ("The rank of the card is: " + str(clf.predict([hog_image])[0]))
        
        obssuitres= cv2.resize(obssuit, (20, 35))
        fd, hog_image = hog(obssuitres, orientations=9, pixels_per_cell=(8, 8), 
                        cells_per_block=(2, 2), visualize=True, multichannel=False)
        hog_image= np.reshape(hog_image, (1, hog_image.shape[0]*hog_image.shape[1]))[0]
        print ("The suit of the card is: " + str(qclf.predict([hog_image])[0]))


# In[21]:


def rankplussuit_detect2(imarray, cntsarray):
    cards= []
    for cnt in cntsarray:
        obsrank= get_rank(imarray, cnt)
        obssuit= get_suit(imarray, cnt)
        rankPath= './templates2/ranks/'
        suitPath= './templates2/suits/'
        ssesuits= []
        sseranks= []
        for fold in os.listdir(rankPath):
            summ= 0
            for img in os.listdir(os.path.join(rankPath, fold)):
                fullname= os.path.join(os.path.join(rankPath, fold), img)
                rank= np.array(cv2.imread(fullname, 0))
                minx= min(rank.shape[0], obsrank.shape[0])
                miny= min(rank.shape[1], obsrank.shape[1])
                obsrankres= cv2.resize(obsrank, (miny, minx))
                rankres= cv2.resize(rank, (miny, minx))
                fd, hog_image = hog(rankres, orientations=9, pixels_per_cell=(8, 8), 
                    cells_per_block=(2, 2), visualize=True, multichannel=False)
                fd2, hog_image2 = hog(obsrankres, orientations=9, pixels_per_cell=(8, 8), 
                    cells_per_block=(2, 2), visualize=True, multichannel=False)
                summ+= np.sum(np.abs(hog_image - hog_image2)**2)
            sseranks.append(summ)
        print (sseranks)
        rp= os.listdir(rankPath)[np.argmin(sseranks)]
        print ("The rank of the card is: " + os.listdir(rankPath)[np.argmin(sseranks)])
        
        for fold in os.listdir(suitPath):
            summ= 0
            flag= False
            for img in os.listdir(os.path.join(suitPath, fold)):
                fullname= os.path.join(os.path.join(suitPath, fold), img)
                suit= np.array(cv2.imread(fullname, 0))
                minx= min(suit.shape[0], obssuit.shape[0])
                miny= min(suit.shape[1], obssuit.shape[1])
                obssuitres= cv2.resize(obssuit, (miny, minx))
                suitres= cv2.resize(suit, (miny, minx))
                fd, hog_image = hog(suitres, orientations=9, pixels_per_cell=(8, 8), 
                    cells_per_block=(2, 2), visualize=True, multichannel=False)
                fd2, hog_image2 = hog(obssuitres, orientations=9, pixels_per_cell=(8, 8), 
                    cells_per_block=(2, 2), visualize=True, multichannel=False)
                summ+= np.sum(np.abs(hog_image - hog_image2)**2)
            ssesuits.append(summ)
        print (ssesuits)
        sp= os.listdir(suitPath)[np.argmin(ssesuits)]
        print ("The suit of the card is: " + os.listdir(suitPath)[np.argmin(ssesuits)])
        cards.append([rp, sp])
    return cards


# In[22]:


imarray= np.array(cv2.imread('./im1.jpeg', 0))
cntsarray= card_detect('./im1.jpeg')
print (len(cntsarray))
cards= rankplussuit_detect2(imarray, cntsarray)


# In[16]:


os.listdir('./templates/suits/')


# In[ ]:




