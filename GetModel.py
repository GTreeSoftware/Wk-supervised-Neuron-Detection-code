"""This function is used to load the VoxResNet model for training or testing"""
import torch.nn as nn
import torch
from model.VoxResNet import VoxResNet_3D

# choose your model: you can have your own model, just add it to the model file and build the model name
model_choice={
    'VoxResNet_3D':VoxResNet_3D,
}

# get the model for training or testing, based on your choice in the config file.
def GetModel(opt):

    if opt.model_choice == 'VoxResNet_3D':
        # act_fn=nn.LeakyReLU(0.01)
        model=model_choice[opt.model_choice](opt.in_dim,opt.out_dim)

    # decide whether to use cuda or not
    if opt.use_cuda:
        model=nn.DataParallel(model).cuda()

    return model





