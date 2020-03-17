"""The common file include the following functions for the network training, testing and saving:
   1.GetDatasetLoader: generate the dataloader for training or validation
   2.GetOptimizer: generate the optimizer for training
   3.GetScheduler: adjust the learning rate of the optimizer parameters
   4.save_parameters: save the parameters of the best parameters of the training model
   5.load_checkpoint: load the checkpoints of the best parameters of the training model
   6.save_img2tiff: save the digital image to the tifffile format
   7.GetWeight: generate the training weight for the foreground and background classes
   8.Logger: this class is used to record the training or validation process log
   9.WriteLog: write the log files
   """

from dataset.Get_Dataset import GetDataset
from torch.utils.data import DataLoader
import torch.optim as optim
from utils.tools import make_dirs
import torch
import sys
import tifffile
import os
import numpy as np
import shutil
import random

# generate the dataloader for training or validation
def GetDatasetLoader(opt,prefix='WeaklySupervised',phase='train',augument=None):
    # the phase can be train or validation
    dataset1 = GetDataset(prefix, phase=phase, augument=augument)

    if phase == 'train':
        dataloader1 = DataLoader(dataset1, batch_size=opt.train_batch, num_workers=opt.num_workers,shuffle=opt.train_shuffile)

    if phase == 'val':
        dataloader1 = DataLoader(dataset1, batch_size=opt.val_batch, num_workers=opt.num_workers, shuffle=False)

    return dataloader1

# generate the optimizer for training
def GetOptimizer(opt,model):
    if opt.optimizer=='SGD':
        optimizer1=optim.SGD(
            model.parameters(),lr=opt.lr,momentum=opt.momentum,
            weight_decay=opt.weight_decay,
        )

    if opt.optimizer=='RMSp':
        optimizer1=optim.RMSprop(
            model.parameters(),lr=opt.lr,alpha=opt.alpha,
            weight_decay=opt.weight_decay,
        )

    return optimizer1

# adjust the learning rate of the optimizer parameters
def GetScheduler(opt,optimizer1):
    scheduler1=optim.lr_scheduler.StepLR(
        optimizer1,step_size=opt.step_size,gamma=opt.gamma
    )
    return scheduler1

# save the parameters of the best parameters of the training model
def save_parameters(state,best_value='VoxResNet'):
    save_path = 'checkpoints'
    make_dirs(save_path)

    # save_name = save_path + '/model_parameters_value{:.3f}.pth'.format(best_value)
    save_name = save_path + '/{}.ckpt'.format(best_value)
    torch.save(state, save_name)

# load the checkpoints of the best parameters of the training model
def load_checkpoint(model, checkpoint_PATH, optimizer):
    model_CKPT = torch.load(checkpoint_PATH)
    model.load_state_dict(model_CKPT['state_dict'])
    print('loading checkpoint!')
    optimizer.load_state_dict(model_CKPT['optimizer'])
    start_epoch = model_CKPT['epoch'] + 1

    return model, optimizer,start_epoch

# save the digital image to the tifffile
def save_img2tiff(img,file_name):
    save_name=file_name
    tifffile.imsave(save_name,img,dtype=np.uint8)
    print('saved:',save_name)

# generate the training weight for the foreground and background classes
def GetWeight(opt,target,slr=0,is_t=1):
    if target.device.type=='cuda':
        beta = target.sum().cpu().numpy().astype(np.float32) / (target.numel() + 1e-5)
    else:
        beta = target.sum().numpy().astype(np.float32) / (target.numel() + 1e-5)

    beta = beta + slr
    weight = np.array([beta, 1 - beta])

    # whether the format is tensor or not
    if is_t:
        weight = torch.tensor(weight)
        if opt.use_cuda:
            weight = weight.float().cuda()

    return weight

# used to record the training or validation process log
# http://stackoverflow.com/questions/34950201/pycharm-print-end-r-statement-not-working
class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout  # stdout
        self.file = None

    def open(self, file, mode=None):
        if mode is None:
            mode = 'w'
        self.file = open(file, mode)

    def write(self, message, is_terminal=1, is_file=1):
        if '\r' in message:
            is_file = 0

        if is_terminal == 1:
            self.terminal.write(message)
            self.terminal.flush()
            # time.sleep(1)

        if is_file == 1:
            self.file.write(message)
            self.file.flush()

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass

# write the log files
def WriteLog(opt1):
    log = Logger()
    make_dirs(opt1.log_path)
    log.open(opt1.log_name, mode='a')
    log.write('** experiment settings **\n')
    log.write('\toptimizer:               {}  \n'.format(opt1.optimizer))
    log.write('\tlearning rate:         {:.3f}\n'.format(opt1.lr))
    log.write('\tweight_decay:         {:.4f}\n'.format(opt1.weight_decay))
    log.write('\tepoches:               {:.3f}\n'.format(opt1.train_epoch))
    log.write('\tpatch_size:              {}  \n'.format(opt1.patch_size))
    log.write('\tmodel:                   {}  \n'.format(opt1.model_choice))
    log.write('\tsave_parameter:                   {}  \n'.format(opt1.save_parameters_name))

    return log


