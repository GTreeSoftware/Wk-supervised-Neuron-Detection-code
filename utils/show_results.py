"""This is used for 3D show of the segmentation result"""

import os
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from utils.tools import make_dirs
import skimage.io as skio

# get the mip of the 3D image along one axis
def Get_MIP_Image(img,direction):
    im1=np.max(img,axis=direction)
    return im1


# get the mip of the 3D image along 3 axises
def Get_Joint_MIP(img):
    if np.max(img)<=1:
        img=np.uint8(img*255)

    im0=np.max(img,0)
    im1 = np.max(img, 1)
    im2 = np.max(img, 2)
    imh = np.hstack([im0, im1, im2])
    return imh


def plot_img_and_mask(img, mask):
    fig = plt.figure()
    a = fig.add_subplot(1, 2, 1)
    a.set_title('Input image')
    plt.imshow(img)

    b = fig.add_subplot(1, 2, 2)
    b.set_title('Output mask')
    plt.imshow(mask)
    plt.show()


# this function is used to adjust a suitable image window for visualization
def ImageForVis(image_recon):
    image_recon = np.array(image_recon, np.int32)
    image_recon[image_recon > 350] = 350
    image_recon[image_recon < 100] = 100
    image_recon = (image_recon - np.min(image_recon)) / (np.max(image_recon) - np.min(image_recon))
    image_recon=np.uint8(image_recon * 255)
    return image_recon





