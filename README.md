##################################################
This belongs to Britton Chance Center for Biomedical Photonics, Wuhan National Laboratory for Optoelectronics-Huazhong University of Science and Technology, Hubei, China
Author: Huang

Weakly supervised learning of 3D deep network for neuron reconstruction

Object:
    This is an implementation of weakly supervised deep learning on Python 3.6 and pytorch based on VoxResNet.
    Specifically, we applied a 3D residual CNN as the architecture for discriminative neuronal features extraction.
    instead of using manual annotations, a weakly supervised learning framework was proposed by iteratively training the CNN model for improved 
    prediction and refining the pseudo labels for training samples updating. The pseudo label was iteratively modified by mining and adding weak 
    neurites from the CNN predicted probability map based on the tubularity and continuity of neurites.  

Requirements
    Python 3.6, pytorch, visdom and other common packages including numpy, tqdm, time, tifffile, glob, skimage, shutil, random, scipy, matplotlib,csv and pprint.

Train supervised deep learning network for neurite segmentation:
    Use the Train_Weakly_Supervised.py File for training. The config.py file is used for parameters setting.
    1. use the Split_Dataset file in the dataset folder to generate your own dataset
    2. if the training is from the begining, set lr parameter in the config file to 0.01.
       if transfer learning is used, set lr parameter in the config file to 0.001 and replace the trained model name in line 107.
    3. run the function for training. 
    You can have your own dataset by change the folder path of your dataset in line 128 in Split_Dataset.py file.
    
Train weakly-supervised deep learning network for neurite segmentation:
    Iteratively using the Train_Weakly_Supervised.py File and Generate_Refined_Samples for pseudo labels refinement and re-training. 
    The config.py file is used for parameters setting.
    The Generate_Refined_Samples.py file is used to refine the pseudo labels for training samples updating by mining and adding weak neurites from the CNN predicted probability map based on the tubularity and continuity of neurites.
    For Train_Weakly_Supervised.py File:
    1. use the Split_Dataset file in the dataset folder to generate your own dataset
    2. if the training is from the begining, set lr parameter in the config file to 0.01.
       if transfer learning is used, set lr parameter in the config file to 0.001 and replace the trained model name in line 107.
    3. run the function for training. 
    You can have your own dataset by change the folder path of your dataset in line 128 in Split_Dataset.py file.
    
    For Generate_Refined_Samples.py File to train your own dataset:
    1. place the folder path of your own training dataset in line 172. Image format: tiff.
    2. place the previously trained network parameters name in line 183.
    3. run the function for pseudo label refinement.

Predict neurite probability map for new images using previously trained model:
    Use Predict_File.py for new dataset prediction.
    usage:
    1. place the folder path of your own dataset in line 78. Image format: tiff
    2. set the saving path for your own dataset in line 82.
    3. place the previously trained network parameters name in line 90.
    4. run the function for prediction

Neuron datasets:
    We also published some neuron images to encourage users to try out. The images can be found in the neurite_data folder.

Attention:
    Users could change some default parameters values based on their own computation power, including batch_size and num_workers.
  

