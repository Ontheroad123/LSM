# ply
import cv2 as cv2
import numpy as np

import math
from os.path import basename, split, join, dirname
from util3 import *
import torch
import scipy as sp
import scipy.ndimage


def copysmallobjects2_dilate(image_dir, label_dir, mask_dir, save_base_dir, save_mask_dir, enlarge, paste_number, dtype):

    image = cv2.imread(image_dir, 0)
    image1 = image.copy()
    liver_mask = np.where(image>0, 1, 0).astype(np.uint8)
    mask = cv2.imread(mask_dir, 0)/255
    mask1 = cv2.imread(mask_dir, 0)
    labels = read_label_txt(label_dir)
    if len(labels) == 0:
        return
    centers, radiuss= rescale_yolo_labels(labels, image.shape)  # 转换坐标表示
    newcenters = centers.copy()

    for i, center  in enumerate(centers):
        print('center', center)
        redius = radiuss[i]
        crop_ori = cropmask(image, center, redius)
        mask_ori = cropmask(mask, center, redius)
        tmp_redius = redius+enlarge
        kernel = np.ones((2*enlarge+1,2*enlarge+1), np.uint8)
        mask_ori_dilate = cv2.dilate(mask_ori, kernel, 1)
       
        #if (mask_ori_dilate*mask).sum()==mask_ori.sum() and mask_ori.sum()<1024:
        new_center = random_add_patches2_dilate(image, liver_mask, mask, redius, newcenters, radiuss, enlarge,  paste_number, iou_thresh=0)
        print(new_center)
        if len(new_center) !=0:
            
            w_c = new_center[0] - center[0]
            h_c = new_center[1] - center[1]
            if dtype =='dilate_no':
                for i in range(center[0]-tmp_redius, center[0]+tmp_redius):
                    for j in range(center[1]-tmp_redius, center[1]+tmp_redius):
                        if mask_ori_dilate[j][i]!=0:
                            ni = j+h_c 
                            nj = i+w_c
                            newpoint = (nj,ni)
                            if image[newpoint[1]][newpoint[0]]!=0 and image[j][i]!=0:
                                image1[newpoint[1]][newpoint[0]] = image[j][i]
                            else:
                                if mask_ori[j][i]!=0:
                                    image1[newpoint[1]][newpoint[0]] = image[j][i]
            elif dtype=='dilate_ave':
                for i in range(center[0]-redius-enlarge, center[0]+redius+enlarge):
                    for j in range(center[1]-redius-enlarge, center[1]+redius+enlarge):
                        if mask_ori_dilate[j][i]!=0:
                            ni = j+h_c 
                            nj = i+w_c
                            newpoint = (nj,ni)
                            if image[newpoint[1]][newpoint[0]]!=0 and image[j][i]!=0:
                                if mask_ori[j][i] ==0:
                                    image1[newpoint[1]][newpoint[0]] =  round(image[j][i]/2)+round(image[newpoint[1]][newpoint[0]]/2)
                                else:
                                    image1[newpoint[1]][newpoint[0]] = image[j][i]
                            elif mask_ori[j][i]!=0:
                                image1[newpoint[1]][newpoint[0]] = image[j][i]
            for i in range(center[0]-redius, center[0]+redius):
                for j in range(center[1]-redius, center[1]+redius):
                    if mask_ori[j][i]!=0:
                        newpoint =[ i+w_c, j+h_c ]
                        mask1[newpoint[1]][newpoint[0]] = 255

            newcenters.append(new_center)
            radiuss.append(tmp_redius)
                
    savename = save_base_dir+'/'+label_dir.split('/')[-1][:-3]+'png'
    print(savename)
    cv2.imwrite(savename, image1)

    savemaskname =  save_mask_dir+'/'+label_dir.split('/')[-1][:-3]+'png'
    print(savemaskname)
    cv2.imwrite(savemaskname, mask1)

