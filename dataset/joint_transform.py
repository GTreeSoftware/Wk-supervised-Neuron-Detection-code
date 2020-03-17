"""
This function is used for data augmentation for dataset
including：random flip, rotation, adding or subtracting Gaussian noise, change the brightness and intensity range of images
"""

import random
from utils.show_results import *

# whole function for all the data augmentations
class JointCompose(object):
    def __init__(self,transforms):
        self.transforms=transforms

    def __call__(self, img, mask):
        assert img.shape == mask.shape
        for t in self.transforms:
            img,mask=t(img,mask)
        return img,mask

# different data augmentation
class JointRandomFlip(object):
    def __call__(self,img,mask):
        p=0.5
        axis = random.randint(1, 2)
        if random.random()<p:
            img=np.flip(img,axis).copy()
            mask=np.flip(mask,axis).copy()
        return img,mask


class JointRandomRotation(object):
    def __call__(self, img, mask):
        ang=random.randint(1,3)
        axis=(1,2)
        p=0.5
        if random.random()<p:
            img=np.rot90(img,ang,axis).copy()
            mask=np.rot90(mask,ang,axis).copy()
        return img,mask


class JointRandomGaussianNoise(object):
    """define the noise amplitude and weight : img float and label long"""
    def __init__(self,amplitude=10):
        self.amplitude=amplitude

    def __call__(self, img, mask):
        # eliminate the std of the image
        nlevel=random.random()*self.amplitude/350
        p=0.5
        if random.random()<p:
            noise=nlevel*np.random.normal(0,1,img.shape)
            img=img+noise

        return img,mask


class JointRandomSubTractGaussianNoise(object):
    """define the noise amplitude and weight : img float and label long"""
    def __init__(self,amplitude=10):
        self.amplitude=amplitude

    def __call__(self, img,mask):
        nlevel=random.random()*self.amplitude/350
        p=0.5
        if random.random()<p:
            noise=nlevel*np.random.normal(0,1,img.shape)
            img=img-noise

        return img,mask


class JointRandomBrightness(object):
    def __init__(self,limit):
        self.limit=limit

    def __call__(self, img,mask):
        p=0.5
        if random.random()<p:
            alpha=1.0+np.random.uniform(self.limit[0],self.limit[1])
            img=alpha*img
        return img,mask


class JointRandomIntensityChange(object):
    def __init__(self,limit):
        self.limit=limit

    def __call__(self, img,mask):
        p=0.5
        if random.random()<p:
            alpha=np.random.uniform(self.limit[0],self.limit[1])
            # print(alpha)
            img=img+alpha
        return img,mask


