"""
This is used to generate the train and validation datasets for the images
Here we first search image patches that satisfied some certain rules.
Patches with very few voxels of positive pseudo labels are removed to eliminate blank patches.
We save the the name of the training or validation datasets in the image_list and label_list.
We also save the positions (upper left position) of the selected patches of the images or labels
in the train or val random_patch_ind.npy files for storage space saving.
Thus, the Split_Dataset file should be first applied to generate the position of selected patches
for training or validation.
Some tips for default settings:
1. training and validation datasets have the same parent folder. Training and validation datasets are
   saved in the 'train' and 'val' folders under the parent folder, respectively.
2. corresponding images and labels share the same filenames, and are placed in the "Image" and "Lable" folder respectively.
3. image format: tiff
4. example: tree structure of the training and validation datasets
        parent_folder
        --train
            --Image
                --001.tif
                ...
            --Label
                --001.tif
                ...
        --val
            --Image
                --001.tif
                ...
            --Label
                --001.tif
                ...
"""

from dataset.tools import *
from utils.tools import make_dirs
import shutil
import tifffile

# read the image and label list of corresponding images, and save the names in the corresponding list_files
def GetImageLabelList_Path(data_dir_root,save_prex):
    image_path=os.path.join(data_dir_root,'image')
    label_path=os.path.join(data_dir_root,'label')

    # image and label should have the corresponding filename (same filename)
    name_list = os.listdir(image_path)

    image_list=[os.path.join(image_path,name_num) for name_num in name_list]
    label_list=[os.path.join(label_path,name_num) for name_num in name_list]

    # save the list name
    ilist_name=save_prex+'_image_list.txt'
    WriteList2Txt(ilist_name,new_image_list)

    llist_name=save_prex+'_label_list.txt'
    WriteList2Txt(llist_name,new_label_list)

    return image_list,label_list


# get the name list of the training or validation datasets
def GetList(prefix,phase1='train'):
    data_path=prefix+'/'

    image_list_name = data_path + phase1 + '_image_list.txt'
    label_list_name = data_path + phase1 + '_label_list.txt'

    image_list = ReadTxt2List(image_list_name)
    label_list = ReadTxt2List(label_list_name)

    return image_list,label_list


# get the random patches
def RandomPatches(image,patch_size):
    w,h,z = image.shape
    pw,ph,pz=patch_size

    # calculate the random patches index
    nw,nh,nz=w-pw,h-ph,z-pz
    iw=random.randint(0,nw-1)
    ih=random.randint(0,nh-1)
    iz=random.randint(0,nz-1)

    # at: different from matlab
    image_patches=image[iw:iw+pw,ih:ih+ph,iz:iz+pz]

    return image_patches,[iw,ih,iz]


# calculate the useful mask numbers
def LabelIndexNum(label_image):
    image1=label_image>0
    image1=image1.flatten()
    sum_index=np.sum(image1)

    return sum_index


# random choose patches that satisfied some rules
def RandomSelectPatches(patch_size,patch_num,prefix,threshold1=900,phase='train'):
    # calculate how many patches to generate
    image_list, label_list = GetList(prefix,phase)

    # begin to generate new patch
    new_patch_index=[]
    count_num=0

    while count_num<patch_num:
        # random get image index
        inum = random.randint(0,len(image_list)-1)

        # random choose a patch
        label1 = tifffile.imread(label_list[inum])
        label_patch1,position1=RandomPatches(label1,patch_size)
        label_patch_indnum = LabelIndexNum(label_patch1)

        # calculate whether the patch satisfies the rule
        if label_patch_indnum>=threshold1:
            new_position=np.hstack((position1,inum))
            new_patch_index.append(new_position)
            count_num+=1

            # save the index list
            np.save(prefix+'/{}_random_patch_ind.npy'.format(phase), new_patch_index)


if __name__=="__main__":
    ####### this is used to generate the datasets for training or validation ######
    parent_dir = 'MyWeaklySupervised'    # parent folder

    ####### here to put your own dataset for training or validation ######
    path_train = parent_dir+'/train'
    path_val = parent_dir+'/val'

    ####### here to generate the name list of the datasets ######
    save_dir = 'WeaklySupervised'  # folder path for list files saving
    make_dirs(save_dir)
    train_prex=save_dir+'/train'
    val_prex=save_dir + '/val'

    GetImageLabelList_Path(path_train, train_prex)
    GetImageLabelList_Path(path_val, val_prex)

    ####### here to generate the selected patches of the datasets ######
    patch_size = opt.patch_size

    # 35 and 15 are train and val patches respectively
    RandomSelectPatches(patch_size, 35*30, save_dir, threshold1=0.0005*120*120*120, phase='train')
    RandomSelectPatches(patch_size, 15*30, save_dir, threshold1=0.0005*120*120*120, phase='val')

    print('ok')





