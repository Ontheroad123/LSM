
import aug3_dilate_gauss as am
import os
from os.path import join
import random
from tqdm import tqdm
import torch
base_dir = os.getcwd()


#segmentation dataset directory and object detection file
tumor = '../demo/data/train'
mask = '../demo/data/trainmask'
txtpath = '../demo/data/labels'
imgs_dir = []
labels_dir = []
masks_dir = []
for fil in os.listdir(tumor):
    tumor1 = os.path.join(tumor, fil)
    for fil1 in os.listdir(tumor1):
        maskpath = mask+'/'+fil+'/'+fil1
        if os.path.exists(maskpath):
            txtname = txtpath+'/train/'+fil+'_'+'.'.join(fil1.split('.')[:-2])+'.txt'
            if os.path.exists(txtname):
                labels_dir.append(txtname)
                masks_dir.append(maskpath)
                imgs_dir.append(tumor1+'/'+fil1)
print(len(imgs_dir))

#外扩的倍数（1，1.5，2）
enlarge = 1
#粘贴次数
paste_number = 1
#augmentation type
dtype='dilate_reverse'
savepath = '../demo/{}_{}_{}'.format(dtype, paste_number, enlarge)
save_base_dir = savepath + '/train'
save_mask_dir = savepath + '/trainmask'
if not os.path.exists(savepath):
    os.mkdir(savepath)
    os.mkdir(save_base_dir)
    os.mkdir(save_mask_dir)

for image_dir, label_dir, mask_dir in tqdm(zip(imgs_dir, labels_dir, masks_dir)):
    am.copysmallobjects2_dilate(image_dir, label_dir, mask_dir, save_base_dir, save_mask_dir, enlarge, paste_number, dtype)
    #break
