#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
cardDict= {'2':2, '3':3, '4':4, '5':5, '6':6, '7':7, '8':8, '9':9, '10':10, 'J':11, 'Q':12, 'K':13, 'A':14}
suitDict= {'Hearts':0, 'Diamond':1, 'Spades':2, 'Clubs':3}


# In[3]:


def check_royal_flush(cards, countmat):
    for i in range(len(countmat)):
        if(sum(countmat[i][-5:])== 5):
            return True
    return False


# In[4]:


def check_straight_flush(cards, countmat):
    for i in range(len(countmat)):
        for j in range(1, 7):
            if(sum(countmat[i][j:j+5])== 5):
                return True
        if(sum(countmat[i][0:4])== 4 and countmat[i][12]== 1):
            return True
    return False


# In[5]:


def check_four_kind(cards, countmat):
    for i in range(len(countmat[0])):
        if(sum(countmat[:, i])== 4):
            return True
    return False


# In[10]:


def check_full_house(cards, countmat):
    for i in range(len(countmat)):
        for j in range(len(countmat)):
            if(sum(countmat[i])== 3 and sum(countmat[j])== 2):
                return True
    return False


# In[8]:


def check_flush(cards, countmat):
    for i in range(len(countmat)):
        if(sum(countmat[i])>= 5):
            return True
    return False


# In[9]:


def check_straight(cards, countmat):
    cardcounts= np.zeros(13)
    for i in range(len(cards)):
        cardcounts[cardDict[cards[i][0]]-2]+= 1
    for i in range(0, 7):
        if(sum(cardcounts[i:i+5])>= 5 and max(cardcounts[i:i+5])-min(cardcounts[i:i+5])<= 1):
            return True
    if(sum(cardcounts[0:4])+cardcounts[12]>= 5 and max(max(cardcounts[0:4]), cardcounts[12])-min(min(cardcounts[0:4]), cardcounts[12])<= 1):
        return True
    return False


# In[2]:


def check_three_kind(cards, countmat):
    for i in range(len(countmat[0])):
        if(sum(countmat[:, i])== 3):
            return True
    return False


# In[1]:


def check_two_pairs(cards, countmat):
    for i in range(len(countmat[0])):
        for j in range(len(countmat[0])):
            if(i!= j and sum(countmat[:, i])== 2 and sum(countmat[:, j]== 2)):
                return True
    return False


# In[6]:


def check_pair(cards, countmat):
    for i in range(len(countmat[0])):
        if(sum(countmat[:, i])== 2):
            return True
    return False


# In[7]:


def check_high_card(cards, countmat):
    counts= np.sum(countmat, axis= 0)
    if(sum(counts[-4:])> 0):
        return True
    return False


# In[11]:


def handDetector(cards):#list of 7 tuples
    countmat= np.zeros((4, 13))
    for i in range(len(cards)):
        countmat[suitDict[cards[i][1]]][cardDict[cards[i][0]]-2]+= 1
    countmat= np.array(countmat)
    print (countmat)
    
    if(check_royal_flush(cards, countmat)):
        return "royal flush"
    if(check_straight_flush(cards, countmat)):
        return "straight flush"
    if(check_four_kind(cards, countmat)):
        return "four of kind"
    if(check_full_house(cards, countmat)):
        return "full house"
    if(check_flush(cards, countmat)):
        return "flush"
    if(check_straight(cards, countmat)):
        return "straight"
    if(check_three_kind(cards, countmat)):
        return "three of kind"
    if(check_two_pairs(cards, countmat)):
        return "two pairs"
    if(check_pair(cards, countmat)):
        return "pair"
    if(check_high_card(cards, countmat)):
        return "high card"
    return "bro just call it please!"

