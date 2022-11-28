
import cv2 as cv2
import numpy as np
from math import e
import math
from os.path import basename, split, join, dirname
from util3 import *
import torch
import scipy as sp
import scipy.ndimage

def gauss(dilate_num,x):
    var = (math.floor(dilate_num/2)+1)**2/(math.log(2,e))
    f = math.exp(-x**2/var)
    return f

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
        redius = radiuss[i]
        crop_ori = cropmask(image, center, redius)
        mask_ori = cropmask(mask, center, redius)
        kernel = np.ones((3,3), np.uint8)

        mask_ori_dilate1 = cv2.dilate(mask_ori, kernel, 1)
        mask_ori_dilate2 = cv2.dilate(mask_ori_dilate1, kernel, 1)
        mask_ori_dilate3 = cv2.dilate(mask_ori_dilate2, kernel, 1)
        mask_ori_dilate4 = cv2.dilate(mask_ori_dilate3, kernel, 1)
        mask_ori_dilate5 = cv2.dilate(mask_ori_dilate4, kernel, 1)
        mask_ori_dilate6 = cv2.dilate(mask_ori_dilate5, kernel, 1)
        mask_ori_dilate7 = cv2.dilate(mask_ori_dilate6, kernel, 1)


        if (mask_ori_dilate7*mask).sum()==mask_ori.sum() and mask_ori.sum()<1024:
            new_center = random_add_patches2_dilate(image, liver_mask, mask, redius, newcenters, radiuss, enlarge,  paste_number=1, iou_thresh=0)
            if len(new_center) !=0:
                w_c = new_center[0] - center[0]
                h_c = new_center[1] - center[1]
                #image1 = np.zeros_like(image)
                for i in range(center[0]-redius-enlarge, center[0]+redius+enlarge):
                    for j in range(center[1]-redius-enlarge, center[1]+redius+enlarge):
                        if mask_ori_dilate5[j][i]!=0:
                            ni = j+h_c 
                            nj = i+w_c
                            newpoint = (nj,ni)
                            if image[newpoint[1]][newpoint[0]]!=0:
                                if mask_ori[j][i] ==0:
                                    dist = 0
                                   
                                    if mask_ori_dilate1[j][i]!=0:
                                        dist = 1
                                    elif mask_ori_dilate2[j][i]!=0:
                                        dist = 2
                                    elif mask_ori_dilate3[j][i]!=0:
                                        dist = 3
                                    elif mask_ori_dilate4[j][i]!=0:
                                        dist = 4
                                    elif mask_ori_dilate5[j][i]!=0:
                                        dist = 5
                                    elif mask_ori_dilate6[j][i]!=0:
                                        dist = 6
                                    elif mask_ori_dilate7[j][i]!=0:
                                        dist = 7
                                   
                                    dis = gauss(enlarge, dist)
                                    if dtype=='dilate_gauss':
                                        image1[newpoint[1]][newpoint[0]] =  round(dis*image[j][i]+(1-dis)*image[newpoint[1]][newpoint[0]])
                                    
                                    elif dtype=='dilate_reverse':
                                        image1[newpoint[1]][newpoint[0]] = round((1-dis)*image[j][i]+dis*image[newpoint[1]][newpoint[0]])
                                else:
                                    image1[newpoint[1]][newpoint[0]] = image[j][i]
                            elif mask_ori[j][i]!=0:
                                image1[newpoint[1]][newpoint[0]] = image[j][i]
                #cv2.imwrite('test1.png', image1)
                for i in range(center[0]-redius, center[0]+redius):
                    for j in range(center[1]-redius, center[1]+redius):
                        if mask_ori[j][i]!=0:
                            newpoint =[ i+w_c, j+h_c ]
                            mask1[newpoint[1]][newpoint[0]] = 255
                newcenters.append(new_center)
                radiuss.append(enlarge+redius)
                
    savename = save_base_dir+'/'+label_dir.split('/')[-1][:-3]+'png'
    print(savename)
    cv2.imwrite(savename, image1)

    savemaskname =  save_mask_dir+'/'+label_dir.split('/')[-1][:-3]+'png'
    print(savemaskname)
    cv2.imwrite(savemaskname, mask1)

