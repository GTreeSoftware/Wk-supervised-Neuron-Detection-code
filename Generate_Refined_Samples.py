"""
This function is used to refine the pseudo labels for training samples updating by mining
and adding weak neurites from the CNN predicted probability map based on the tubularity and continuity of neurites.
usage:
1. place the folder path of your own training dataset in line 172. Image format: tiff.
2. place the previously trained network parameters name in line 183.
3. run the function for pseudo label refinement.

The Train_Weakly_Supervised file and Generate_Refined_Samples file are iteratively used for pseudo labels refinement.
"""

from Common import *
from GetModel import *
from utils.tools import *
from utils.vis_tool import Visualizer
from utils.meters import AverageMeter
import torch.nn as nn
import torch
from tqdm import tqdm
from config import opt
import time
from utils.eval_metric import ConfusionMeter
import os
import numpy as np
import tifffile
from glob import glob
from dataset.generatePatches import *
from torch.utils.data import DataLoader
import skimage.io as skio
from skimage.morphology import closing,binary_closing,ball,skeletonize_3d,binary_dilation
from skimage import measure
from skimage import morphology
import skimage.util
import warnings

warnings.filterwarnings('ignore')

# remove small objects for denoising
def denoise_by_connection(image, min_size):
    image = image > 0.8
    image = image.astype(np.bool)
    image_new = morphology.remove_small_objects(image, min_size, connectivity=2)
    image_new = image_new.astype(np.int)
    image_new = image_new > 0.8
    image_new = image_new.astype(np.int)
    return image_new


def Calculate_Nearby_Intesnity_Diff(image1,seg1,radius1):
    seg1=seg1>0
    roi_expansion=binary_dilation(seg1,ball(radius1))
    roi_expansion[seg1>0]=0
    nearby_mean=np.mean(image1[roi_expansion])
    return nearby_mean


def Adaptive_Threshold_For_Prob(prob1,distance=4):
    # calculate the mean of the intensity without pred_5
    pred_5 = prob1 >= 0.5
    nearby_mean = Calculate_Nearby_Intesnity_Diff(prob1, pred_5, distance)
    return nearby_mean


def Get_ROI_Expension_For_LowThreshold(pred_5_denoise,pred_1):
    # get the expension of the previous roi, and get the final result
    image_shape=pred_5_denoise.shape

    # calculate how many smaples and get the samples with
    pred_5_denoise=pred_5_denoise>0
    pred_1 = pred_1 > 0

    label_image5, label_num5 = measure.label(pred_5_denoise, neighbors=8, return_num=True)
    label_image1, label_num1 = measure.label(pred_1, neighbors=8, return_num=True)

    # begin for searching
    out_result=np.zeros(image_shape)
    out_result=out_result.astype(np.bool)

    if label_num5>0:
        # get the output of the final result
        for i in range(label_num5):
            new_image5=label_image5==(i+1)
            new_image5=new_image5>0

            # just get one seed
            index5=np.where(new_image5>0)

            seedx=index5[0][1]
            seedy=index5[1][1]
            seedz=index5[2][1]

            # get the expanded area of the image
            label_ind1=label_image1[seedx,seedy,seedz]
            new_image1=(label_image1==label_ind1)
            out_result[new_image1>0]=1
    return out_result


def Process_For_Adaptive_result(patch_size,overlap,pred_recon,prob_recon, distance=4,denoise_size=400):
    # split the dataset for new one and get the final result
    # begin to generate patches
    patchindices = compute_patch_indices(pred_recon.shape, patch_size, overlap, start=0)

    pred_Ada_patches=[]

    nearby_mean = Adaptive_Threshold_For_Prob(prob_recon, distance=distance)

    for ind in range(patchindices.shape[0]):
        pred_patch=get_patch_from_3d_data(pred_recon, patch_shape=patch_size, patch_index=patchindices[ind, :])
        pred_patch=pred_patch>0

        prob_patch=get_patch_from_3d_data(prob_recon, patch_shape=patch_size, patch_index=patchindices[ind, :])

        if np.sum(pred_patch)>0:
            pred_patch1 = prob_patch >= nearby_mean
            pred_mpatch1= Get_ROI_Expension_For_LowThreshold(pred_patch, pred_patch1)
            pred_Ada_patches.append(pred_mpatch1)

        else:
            pred_mpatch1=pred_patch
            pred_Ada_patches.append(pred_mpatch1)

    # reconstruct the result
    pred_reconA=reconstruct_from_patches(pred_Ada_patches, patchindices, pred_recon.shape)
    pred_reconA =pred_reconA>0
    pred_reconA=denoise_by_connection(pred_reconA,denoise_size)

    return pred_reconA


def test_model(val_loader,model):
    model.eval()

    pred_Patches = []
    prob_patches =[]

    soft_max = nn.Softmax(dim=1)

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

    return pred_Patches,prob_patches



if __name__=="__main__":
    # place your own file folder path for the training dataset
    file_path = 'MyWeaklySupervised'
    file_names = glob(os.path.join(file_path, 'image/*.tif'))

    # could just use the label path
    recon_path = file_path+'/label'
    # make_dirs(recon_path)

    # begin to predict
    model = GetModel(opt)

    # place the previously trained network parameters
    parameters_name = 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'
    model_CKPT = torch.load(parameters_name)
    model.load_state_dict(model_CKPT['state_dict'])
    print('loading checkpoint!')

    for image_num in range(len(file_names)):
        file_name1 = file_names[image_num]
        save_num = file_name1.split('/')[-1].split('.')[0]

        # read the image and process
        image =tifffile.imread(file_name1)

        # begin to generate patches
        patch_size = np.array([120, 120, 120])
        overlap = np.array([10, 10, 10])

        Tdataset = GenerateDataset_ForNew(image, image.shape, patch_size, overlap)
        val_loader = DataLoader(Tdataset, batch_size=4, num_workers=5, shuffle=False)

        pred_Patches, prob_patches = test_model(val_loader, model)

        # begin to combine the images into one
        patchindices = compute_patch_indices(image.shape, patch_size, overlap, start=0)

        # 0 and 1 recon
        pred_recon = reconstruct_from_patches(pred_Patches, patchindices, image.shape)
        pred_recon = pred_recon.astype(np.uint8)

        # prob recon (need)
        prob_recon = reconstruct_from_patches(prob_patches, patchindices, image.shape)

        ## postprocessing  and save the result
        pred_recon = pred_recon > 0
        pred_recon = denoise_by_connection(pred_recon, 200)

        ###### begin to process the dataset: for fast calculation using split and combine operation
        ## split and combine to get the final result
        patch_size = np.array([120, 120, 120])
        overlap = np.array([10, 10, 10])
        pred_reconA = Process_For_Adaptive_result(patch_size, overlap, pred_recon, prob_recon, distance=4,
                                                  denoise_size=200)

        pred_mrecon1 = binary_closing(pred_reconA, ball(4))
        pred_mrecon1 = skeletonize_3d(pred_mrecon1)
        pred_mrecon1 = pred_mrecon1 > 0
        pred_mrecon1 = binary_dilation(pred_mrecon1, ball(2))

        tifffile.imsave(os.path.join(recon_path, save_num + '.tif'), np.uint8(pred_mrecon1 * 255))

        print('ok')


















































