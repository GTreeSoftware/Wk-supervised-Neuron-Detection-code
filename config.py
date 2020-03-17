"""
This function is used to generate the main parameters for training and validation of the model
"""

from pprint import pprint
import torch
import numpy as np
import os
import time

class Config:
    ############################## input dataset files #########################
    dataset_prefix='WeaklySupervised_Train'

    ############################# input parameters for data #####################
    # 1. training or validation patch
    patch_size = [120,120,120]
    pixel_size=[0.2,0.2,1]

    # 2. training choice
    train_augument=True
    train_shuffile=True

    train_epoch=100
    train_plotfreq=2

    # 3. validation choice
    val_run=True
    val_plotfreq=2

    save_img = True

    # 4. test choice
    image_size = [300, 300, 300]
    overlap = 10
    patch_valnum = np.ceil(image_size[0] / (patch_size[0] - overlap)) ** 3

    ################### input parameters for the computation #####################
    # you can change the parameters according to the configuration of your computerS
    num_workers = 5
    use_cuda = True
    thred = 32

    train_batch = 3
    val_batch = 3

    ################### input parameters for VoxResNet model #####################
    model_choice='VoxResNet_3D'
    in_dim=1
    out_dim=2

    ######################## input parameters for optimizer ######################
    optimizer='SGD'
    lr=0.01
    # lr = 0.001
    momentum=0.9
    weight_decay=0.0005

    scheduler='StepLR'
    step_size=4
    gamma=0.5

    #################### input parameters for result saving ######################
    log_path='result'
    save_parameters_name = 'Weakly_Sup0'
    log_name = os.path.join(log_path, 'log_{}.txt'.format(save_parameters_name))

    #################### transfer learning or not ######################
    load_state = False

    #################### print the information of the configuration ###############
    def _parse(self):
        print('======user config========')
        pprint(self._state_dict())
        print('==========end============')

    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in Config.__dict__.items()
                if not k.startswith('_')}


#################### build the instance ###############
opt = Config()

