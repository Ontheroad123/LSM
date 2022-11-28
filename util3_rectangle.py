import os
import cv2
import numpy as np
from os.path import join, split
import random


def read_label_txt(label_dir):
    labels = []
    with open(label_dir) as fp:
        for f in fp.readlines():
            labels.append(f.strip().split(' '))
    return labels


def rescale_yolo_labels(labels, img_shape, enlarge):
    height, width = img_shape
    rescale_boxes = []
    rescale_boxes_crop = []
    for box in list(labels):
        x_c = float(box[1]) * width
        y_c = float(box[2]) * height
        w = float(box[3]) * width
        h = float(box[4]) * height
        
        x_left = x_c - w /2-1
        y_left = y_c - h/2-1
        x_right = x_c + w /2+1
        y_right = y_c + h /2+1
        rescale_boxes.append([box[0], round(x_left), round(y_left), round(x_right), round(y_right)])

        x_left = x_c - w /2*enlarge
        y_left = y_c - h/2*enlarge
        x_right = x_c + w /2*enlarge
        y_right = y_c + h /2*enlarge
        rescale_boxes_crop.append([box[0], round(x_left), round(y_left), round(x_right), round(y_right)])

    return rescale_boxes, rescale_boxes_crop


def bbox_iou(box1, box2):
    cl, b1_x1, b1_y1, b1_x2, b1_y2 = box1
    cl, b2_x1, b2_y1, b2_x2, b2_y2 = box2
    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = max(b1_x1, b2_x1)
    inter_rect_y1 = max(b1_y1, b2_y1)
    inter_rect_x2 = min(b1_x2, b2_x2)
    inter_rect_y2 = min(b1_y2, b2_y2)
    # Intersection area
    inter_width = inter_rect_x2 - inter_rect_x1 + 1
    inter_height = inter_rect_y2 - inter_rect_y1 + 1
    if inter_width > 0 and inter_height > 0:  # strong condition
        inter_area = inter_width * inter_height
        # Union Area
        b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
        b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)
        iou = inter_area / (b1_area + b2_area - inter_area)
    else:
        iou = 0
    return iou


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
    x1 = min(box[0][0], box[3][0])
    y1 = min(box[0][1], box[1][1])
    x2 = max(box[2][0], box[3][0])
    y2 = max(box[0][1], box[3][1])
    return x1,y1,x2,y2


def random_add_patches2(bbox_img, rescale_boxes_crop, image, mask, enlarge, paste_number, iou_thresh):

    bbox_h, bbox_w = bbox_img
    center_search_space = waijie_juxing1(image)
    x1 = center_search_space[0]
    y1 = center_search_space[1]
    x2 = center_search_space[2]
    y2 = center_search_space[3]
    new_bboxes = []
    new_bboxes_crop = []
    if y2-y1>=bbox_h and x2-x1>=bbox_w:
        
        success_num = 0
        cl = 1
        flag = 0
        while success_num < paste_number:
            #随即选择点作为粘贴矩形的左上角坐标
            new_bbox_x_center, new_bbox_y_center = norm_sampling1(center_search_space, image)   # 随机生成点坐�?            
            flag+=1
            if flag>1000:
                break
            new_h = round(enlarge*bbox_h)
            new_w = round(enlarge*bbox_w)
            if round(enlarge*bbox_h)/2==1:
                new_h = round(enlarge*bbox_h)+1
            if round(enlarge*bbox_w)/2==1:
                new_w = round(enlarge*bbox_w)+1
            crop_tmp = mask[new_bbox_y_center:new_h+new_bbox_y_center, new_bbox_x_center:new_w+new_bbox_x_center] 

            if crop_tmp.sum()< (bbox_h*bbox_w*enlarge*enlarge*0.5):
                continue
            if new_bbox_x_center + enlarge*bbox_w > center_search_space[2] :
                continue
            if new_bbox_y_center + enlarge*bbox_h >center_search_space[3]:
                continue
            
            new_bbox_x_left, new_bbox_y_left, new_bbox_x_right, new_bbox_y_right = new_bbox_x_center, \
                                                                                new_bbox_y_center, \
                                                                                new_bbox_x_center + bbox_w, \
                                                                                new_bbox_y_center + bbox_h   
            new_bbox = [cl, round(new_bbox_x_left), round(new_bbox_y_left), round(new_bbox_x_right), round(new_bbox_y_right)]

            new_bbox_x_left, new_bbox_y_left, new_bbox_x_right, new_bbox_y_right = new_bbox_x_center, \
                                                                                new_bbox_y_center , \
                                                                                new_bbox_x_center +  new_w, \
                                                                                new_bbox_y_center +  new_h   
            new_bbox_crop =  [cl, round(new_bbox_x_left), round(new_bbox_y_left), round(new_bbox_x_right), round(new_bbox_y_right)]
            ious = [(bbox_iou(new_bbox_crop, bbox_t)) for bbox_t in rescale_boxes_crop]
            ious2 = [(bbox_iou(new_bbox_crop,bbox_t1)) for bbox_t1 in new_bboxes_crop]
            if ious2 == []:
                ious2.append(0)
            if max(ious) <= iou_thresh and max(ious2) <= iou_thresh:
                success_num += 1
                new_bboxes.append(new_bbox)
                new_bboxes_crop.append(new_bbox_crop)
            else:
                continue

    return new_bboxes, new_bboxes_crop
    
