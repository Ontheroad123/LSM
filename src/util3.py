import os
import cv2
import numpy as np
from os.path import join, split
import random
import math
from math import pi 
random.seed(41)


def read_label_txt(label_dir):
    labels = []
    with open(label_dir) as fp:
        for f in fp.readlines():
            labels.append(f.strip().split(' '))
    return labels


def rescale_yolo_labels(labels, img_shape):
    height, width = img_shape
    centers = []
    radiuss = []
    for box in list(labels):
        x_c = float(box[1]) * width
        y_c = float(box[2]) * height
        w = float(box[3]) * width
        h = float(box[4]) * height
        radius = math.ceil(max(w, h )/2)+1
        center = (int(x_c), int(y_c))
        centers.append(center)
        radiuss.append(radius)
    return centers, radiuss


def cropmask(image,center, rediuns):
    roi = cv2.circle(np.zeros_like(image), center, rediuns, 1, cv2.FILLED)
    roi = image * roi
    return roi


def mask_iou(paste_mask, mask):
    return (paste_mask*mask).sum()


def norm_sampling1(search_space, image):
    flag = 0
    while flag <1:
        search_x_left, search_y_left, search_x_right, search_y_right = search_space
        
        new_bbox_x_center = random.randint(search_x_left, search_x_right)
        new_bbox_y_center = random.randint(search_y_left, search_y_right)
        if image[new_bbox_y_center][new_bbox_x_center] != 0:
            flag +=1
    return [new_bbox_x_center, new_bbox_y_center]


def waijie_juxing1(img):    
    conts, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = sorted(conts, key=cv2.contourArea, reverse=True)[0]
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    
    box = np.int0(box)
    #print('waijie point:', box)
    x1 = min(box[0][0], box[3][0])
    y1 = min(box[0][1], box[1][1])
    x2 = max(box[2][0], box[3][0])
    y2 = max(box[0][1], box[3][1])
    return x1,y1,x2,y2


def random_add_patches2(image, liver_mask, mask,  redius, centers, rediuss, enlarge, paste_number, iou_thresh):
    tmp_redius = round(enlarge*redius)
    
    center_search_space = waijie_juxing1(image)
    x1 = center_search_space[0]
    y1 = center_search_space[1]
    x2 = center_search_space[2]
    y2 = center_search_space[3]
    new_center = []
    new_bboxes_crop = []
    if y2-y1>=redius and x2-x1>=redius:
        
        success_num = 0
        cl = 1
        flag = 0
        while success_num < paste_number:
            #在肝脏的外接矩形中随即选择点作为粘贴的中心点，保证中心点落在肝脏上
            new_bbox_center = norm_sampling1(center_search_space, image)   # 随机生成点坐�?            
            flag+=1
            if flag>1000:
                break
            paste_mask = cropmask(liver_mask, new_bbox_center, tmp_redius)
            #判断粘贴的位置，保证至少有一半落在肝脏上
            if (paste_mask*mask).sum() != 0:
                continue
            if paste_mask.sum()< pi*(tmp_redius)**2/2:
                continue
            if new_bbox_center[0] + redius > center_search_space[2] :
                continue
            if new_bbox_center[1] + redius >center_search_space[3]:
                continue  
            
            #保证粘贴病灶和原始病灶不重叠
            ious = [mask_iou(cropmask(liver_mask, center, round(redius1*enlarge)), paste_mask) for center, redius1 in zip(centers, rediuss)]

            if max(ious) <= iou_thresh:
                success_num += 1
                new_center.append(new_bbox_center)
            else:
                continue
    return new_center
    

def random_add_patches2_dilate(image, liver_mask, mask,  redius, centers, rediuss, enlarge, paste_number, iou_thresh):
    tmp_redius = enlarge+redius
    center_search_space = waijie_juxing1(image)
    x1 = center_search_space[0]
    y1 = center_search_space[1]
    x2 = center_search_space[2]
    y2 = center_search_space[3]
    new_center = []
    new_bboxes_crop = []
    if y2-y1>=redius and x2-x1>=redius:
        
        success_num = 0
        cl = 1
        flag = 0
        while success_num < paste_number:
            new_bbox_center = norm_sampling1(center_search_space, image)   # 随机生成点坐�?            
            flag+=1
            if flag>1000:
                break
            paste_mask = cropmask(liver_mask, new_bbox_center, tmp_redius)
            if (paste_mask*mask).sum() != 0:
                continue
          
            if paste_mask.sum()< pi*tmp_redius**2/2:
                continue
            if new_bbox_center[0] + redius > center_search_space[2] :
                continue
            if new_bbox_center[1] + redius >center_search_space[3]:
                continue  
            
            ious = [mask_iou(cropmask(liver_mask, center, round(enlarge+redius1)), paste_mask) for center, redius1 in zip(centers, rediuss)]

            if max(ious) <= iou_thresh:
                success_num += 1
                new_center = new_bbox_center
            else:
                continue
    return new_center