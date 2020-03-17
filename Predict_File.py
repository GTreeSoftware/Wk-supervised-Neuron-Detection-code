"""
This function is used to get the predicted result of new images using the previously trained network model
usage:
1. place the folder path of your own dataset in line 78. Image format: tiff
2. set the saving path for your own dataset in line 82.
3. place the previously trained network parameters name in line 90.
4. run the function for prediction
"""

from Common import *
from GetModel import *
from utils.tools import *
from utils.meters import AverageMeter
import torch.nn as nn
import torch
import time
from utils.eval_metric import ConfusionMeter
import os
import numpy as np
from dataset.generatePatches import *
from torch.utils.data import DataLoader
from utils.show_results import *
import skimage.io as skio
from glob import glob
from dataset.tools import *
import tifffile


# predict results for new images using previously trained model
def test_model(val_loader,model):
    model.eval()

    pred_Patches = []
    prob_patches =[]

    soft_max = nn.Softmax(dim=1)

    start_time=time.time()
    for batch_ids, (image_patch) in enumerate(val_loader):
        if opt.use_cuda:
            image_patch=image_patch.cuda()
            output=model(image_patch)

            with torch.no_grad():
                # just 0 and 1
                _,pred_patch=torch.max(output,dim=1)

                # for prob
                prob_patch = soft_max(output)
                prob_patch=prob_patch[:,1,...]

                del output

                pred_patch=pred_patch.cpu().numpy()
                prob_patch = prob_patch.cpu().numpy()

                for id1 in range(pred_patch.shape[0]):
                    # 0 and 1
                    pred1=np.array(pred_patch[id1,:,:,:], dtype=np.float32)
                    pred_Patches.append( pred1 )

                    # prob
                    prob1 = np.array(prob_patch[id1, :, :, :], dtype=np.float32)
                    prob_patches.append(prob1)

    # save the images into npy type
    run_time=time.time()-start_time
    print(run_time)

    return pred_Patches,prob_patches


##################################################################
####### This is used for new dataset prediction ##########################
##################################################################
if __name__=="__main__":
    # place the folder path of your own dataset
    file_path='XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'
    file_names = glob(os.path.join(file_path, '*.tif'))

    # set the saving path for your own dataset
    recon_path='XXXXXXXXXXXXXXXXXXXXXXXx'
    recon_path_pred=recon_path+'/pred'
    make_dirs(recon_path_pred)

    # begin to predict
    model = GetModel(opt)

    # place the previously trained network parameters
    parameters_name = 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'
    model_CKPT = torch.load(parameters_name)
    model.load_state_dict(model_CKPT['state_dict'])
    print('loading checkpoint!')

    for image_num in range(len(file_names)):
        file_name1 = os.path.join(file_path,file_names[image_num])
        save_num=file_name1.split('/')[-1].split('.')[0]

        # read the image and process
        image=tifffile.imread(file_name1)

        # begin to generate patches
        patch_size=np.array([120,120,120])
        overlap=np.array([10,10,10])
        patch_indices=compute_patch_indices(image.shape, patch_size, overlap, start=0)

        Tdataset = GenerateDataset_ForNew(image, image.shape, patch_size, overlap)
        val_loader = DataLoader(Tdataset, batch_size=4, num_workers=5, shuffle=False)

        pred_Patches,prob_patches=test_model(val_loader, model)

        patchindices = compute_patch_indices(image.shape, patch_size, overlap,start=0)
        pred_recon = reconstruct_from_patches(pred_Patches, patchindices, image.shape)
        pred_recon=pred_recon.astype(np.uint8)

        # save the output result
        tifffile.imsave(os.path.join(recon_path_pred, save_num + '_pred0.tif'), np.uint8(pred_recon * 255))

        print('ok')

































