#rectangle
import cv2 as cv2
import numpy as np
# from PIL import Image
import random
import math
from os.path import basename, split, join, dirname
from util3_rectangle import *
import torch
from PIL import Image
import scipy as sp
import scipy.ndimage


def cropmask(image,rescale_label):
    x1 = rescale_label[1]
    y1 = rescale_label[2]
    x2 = rescale_label[3]
    y2 = rescale_label[4]
    roi = image[y1:y2, x1:x2]
    return roi

def cropmask1(image, rescale_label):
    x1 = rescale_label[1]
    y1 = rescale_label[2]
    x2 = rescale_label[3]
    y2 = rescale_label[4]
    roi = image[y1:y2, x1:x2]
    newimage = np.zeros_like(image)
    newimage[y1:y2, x1:x2] = roi
    return newimage


def copysmallobjects2_rectangle(image_dir, label_dir, mask_dir, save_base_dir, save_mask_dir, enlarge, paste_number):
   
    image = cv2.imread(image_dir, 0)
    image1 = image.copy()
    liver_mask = np.where(image>0, 1, 0).astype(np.uint8)
    mask = cv2.imread(mask_dir, 0)/255
    mask1 = cv2.imread(mask_dir, 0)
    labels = read_label_txt(label_dir)
    if len(labels) == 0:
        return
    #复制区域最小外接矩形坐标和边长放大enlarge倍后的最小外接矩形坐标
    rescale_labels,  rescale_labels_crop = rescale_yolo_labels(labels, image.shape, enlarge)  # 转换坐标表示
    
    for i, rescale_label  in enumerate(rescale_labels):
        tmp_dirs = rescale_labels_crop[i]
        ori_dirs = rescale_labels[i]
        ori_mask = mask[ori_dirs[2]:ori_dirs[4], ori_dirs[1]: ori_dirs[3] ]
        crop_mask = mask[tmp_dirs[2]:tmp_dirs[4], tmp_dirs[1]: tmp_dirs[3]]
        
        if crop_mask.sum() ==ori_mask.sum()  and  ori_mask.sum()<1024:
            
            roi = cropmask(image, rescale_label)
            #粘贴区域坐标和边长扩大enlarge倍后粘贴坐标
            new_bboxess,  new_bboxes_crops = random_add_patches2(roi.shape, rescale_labels_crop, image, liver_mask, enlarge, paste_number, iou_thresh=0)
            if len(new_bboxess)>0:
                print('newbox:', new_bboxess, new_bboxes_crops)
                for k in range(len(new_bboxess)):
                    new_bboxes = new_bboxess[k]
                    new_bboxes_crop = new_bboxes_crops[k]
                    crop_roi = cropmask1(image, tmp_dirs)
                    little_mask = cropmask1(mask, ori_dirs)
                    
                    center = [round((rescale_label[1]+rescale_label[3])/2), round((rescale_label[2]+rescale_label[4])/2)]
                    new_center = [round((new_bboxes[1]+new_bboxes[3])/2), round((new_bboxes[2]+new_bboxes[4])/2)]
                    w_c = new_center[0] - center[0]
                    h_c = new_center[1] - center[1]
                    #image1 = np.zeros_like(image)
                    for i in range(tmp_dirs[1],tmp_dirs[3]+1):
                        for j in range(tmp_dirs[2], tmp_dirs[4]+1):
                            if crop_roi[j][i]!=0:
                                ni = j+h_c 
                                nj = i+w_c
                                point = (nj,ni)
                                newpoint = point
                                if image[newpoint[1]][newpoint[0]]!=0 and image[j][i]!=0:
                                    image1[newpoint[1]][newpoint[0]] = crop_roi[j][i]
                                elif little_mask[j][i]!=0:
                                    image1[newpoint[1]][newpoint[0]] = crop_roi[j][i]
                    #cv2.imwrite('test.png', image1)
                    
                    for i in range(tmp_dirs[1],tmp_dirs[3]+1):
                        for j in range(tmp_dirs[2], tmp_dirs[4]+1):
                            if little_mask[j][i]!=0:
                                ni = i+w_c
                                nj = j+h_c 
                                point = (ni,nj)
                                newpoint = point
                                mask1[newpoint[1]][newpoint[0]] = 255
                
                    rescale_labels_crop.append(new_bboxes_crop)

    savename = save_base_dir+'/'+label_dir.split('/')[-1][:-3]+'png'
    print(savename)
    cv2.imwrite(savename, image1)

    savemaskname =  save_mask_dir+'/'+label_dir.split('/')[-1][:-3]+'png'
    print(savemaskname)
    cv2.imwrite(savemaskname, mask1)


