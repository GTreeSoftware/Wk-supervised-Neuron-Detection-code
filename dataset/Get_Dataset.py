"""This fucntion is used to get the dataloader for the train and validation datasets
    1. image format: tiff file. you can also get your own dataset by changing the way of image reading
    2. both the train and validation datasets have the corresponding images and labels.
    (Whether the labels are annotated by manual or automatic algorithms)
    3. images are normalized and data augmentation is allowed
"""

import numpy as np
import os
from config import opt
from dataset.tools import *
import random
import torch
from torch.utils.data import DataLoader
from utils.tools import make_dirs
from dataset.joint_transform import *
import tifffile

## this format is used to get the dataset generated by Split_Dataset file
class GetDataset():
    def __init__(self,prefix,phase='train',augument=None):
        # generate the list name of the file. the list name is generated by Split_Dataset file
        module_path = os.path.dirname(__file__)
        data_path=module_path+'/'+prefix+'/'

        image_list_name=data_path+phase+'_image_list.txt'
        label_list_name=data_path+phase+'_label_list.txt'

        self.image_list=ReadTxt2List(image_list_name)
        self.label_list=ReadTxt2List(label_list_name)

        # get the patch index of the dataset
        index_list_name=data_path+phase+'_random_patch_ind.npy'
        self.index_list=np.load(index_list_name)

        # use data augmentation or not
        self.augument=augument

        # used for data normalization
        # for fMOST dataset, the mean value is 150 and to normalize the dataset in range
        # [150 500] (main dynamic range of the data) to [0 1], std1 is set to 350
        self.mean1=np.array([150],dtype=np.float32)
        self.std1=np.array([350],dtype=np.float32)

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, ind):
        # load the patches of the corresponding image and label
        image_num = self.index_list[ind, 3]
        image=tifffile.imread(self.image_list[image_num])
        label=tifffile.imread(self.label_list[image_num])

        # get the patches of the image and label
        patch_position=self.index_list[ind,:3]
        image_patch = GetPatches(image,opt.patch_size,patch_position)
        label_patch = GetPatches(label,opt.patch_size,patch_position)

        # set the data format and normalize the data
        image_patch=np.array(image_patch,dtype=np.float32)
        label_patch=np.array(label_patch,dtype=np.int32)

        # image patch normalization
        image_patch=(image_patch-self.mean1)/self.std1
        if np.max(label_patch)>1:
            label_patch=label_patch/np.max(label_patch)

        # data augmentation
        if self.augument:
            image_patch,label_patch=self.augument(image_patch,label_patch)

        # Expand the dim of the dataset and turn the dataset into torch
        image_patch=np.expand_dims(image_patch,axis=0)

        image_patch=torch.from_numpy(image_patch).float()
        label_patch=torch.from_numpy(label_patch).long()

        return image_patch,label_patch



