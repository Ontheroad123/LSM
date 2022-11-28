#circle
import cv2 as cv2
import numpy as np
import random
import math
from os.path import basename, split, join, dirname
from util3 import *
import torch
from PIL import Image
import scipy as sp
import scipy.ndimage


def copysmallobjects2_circle(image_dir, label_dir, mask_dir, save_base_dir, save_mask_dir, enlarge, paste_number):

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
        redius = radiuss[i]
        crop_ori = cropmask(image, center, redius)
        orimask = cropmask(mask, center, redius)
        
        tmp_redius = round(redius*enlarge)
        orimask1 = cropmask(mask, center, tmp_redius)
        crop_roi = cropmask(image, center, tmp_redius)

        #if (orimask1*mask).sum()==orimask.sum() and orimask.sum()<1024:
        new_centers = random_add_patches2(image, liver_mask, mask, redius, newcenters, radiuss, enlarge,  paste_number, iou_thresh=0)
        if len(new_centers) !=0:
            print('new center:', new_centers)
            for new_center in new_centers:
                w_c = new_center[0] - center[0]
                h_c = new_center[1] - center[1]
                for i in range(center[0]-round(enlarge*redius),center[0]+round(enlarge*redius)):
                    for j in range(center[1]-round(enlarge*redius),center[1]+round(enlarge*redius)):
                        if crop_roi[j][i]!=0:
                            ni = j+h_c 
                            nj = i+w_c
                            newpoint = (nj,ni)
                            if image[newpoint[1]][newpoint[0]]!=0 and image[j][i]!=0:
                                image1[newpoint[1]][newpoint[0]] =  crop_roi[j][i]
                            elif crop_ori[j][i] != 0:
                                image1[newpoint[1]][newpoint[0]] = crop_ori[j][i]
            

                for i in range(center[0]-redius,center[0]+redius):
                    for j in range(center[1]-redius,center[1]+redius):
                        if orimask[j][i]!=0:
                            newpoint =[ i+w_c, j+h_c ]
                            mask1[newpoint[1]][newpoint[0]] = 255
                newcenters.append(new_center)
                radiuss.append(tmp_redius)


    savename = save_base_dir+'/'+label_dir.split('/')[-1][:-3]+'png'
    cv2.imwrite(savename, image1)

    savemaskname =  save_mask_dir+'/'+label_dir.split('/')[-1][:-3]+'png'
    print(savemaskname)
    cv2.imwrite(savemaskname, mask1)

