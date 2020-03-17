"""
This function is used to set some usually used functions
including:
WriteList2Txt: write the list of data into txt format
ReadTxt2List: read the data of txt into a list
GetPatches: get the patch from the given position
Normalization: normalize the data into range [0,1]
"""
import numpy as np
import scipy.io as io
import random
from config import opt
from utils.tools import *
import sys
from glob import glob
import tifffile

# save the file list into txt
def WriteList2Txt(name1,ipTable,mode='w'):
    with open(name1,mode=mode) as fileObject:
        for ip in ipTable:
            fileObject.write(ip)
            fileObject.write('\n')


# Read the file list
def ReadTxt2List(name1,mode='r'):
    result=[]
    with open(name1,mode=mode) as f:
        data = f.readlines()   #read all the trsing into data
        for line in data:
            word = line.strip()  # list
            result.append(word)
    return result


def GetPatches(image,patch_size,position):
    pw,ph,pz=patch_size
    iw,ih,iz=position
    image_patches=image[iw:iw+pw,ih:ih+ph,iz:iz+pz]

    return image_patches


def Normalization(img):
    img=np.array(img,np.float)
    return (img-np.min(img))/(np.max(img)-np.min(img))

