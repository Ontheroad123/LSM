#oricp(1,2,3,4) , rotate, scale
import cv2 as cv2
import numpy as np
import math
from os.path import basename, split, join, dirname
from util3 import *
import torch
import scipy as sp
import scipy.ndimage

#oricp
def copysmallobjects2_oricp(image_dir, label_dir, mask_dir, save_base_dir, save_mask_dir, enlarge, paste_number):

    image = cv2.imread(image_dir, 0)
    image1 = image.copy()
    liver_mask = np.where(image>0, 1, 0).astype(np.uint8)
    mask = cv2.imread(mask_dir, 0)
    mask1 = mask.copy()
    labels = read_label_txt(label_dir)
    if len(labels) == 0:
        return
    #将yolov5坐标转为中心和长半径形式
    centers, radiuss= rescale_yolo_labels(labels, image.shape)  # 转换坐标表示
    newcenters = centers.copy()
   
    for i, center  in enumerate(centers):
        redius = radiuss[i]
        #裁减复制区域
        crop_ori = cropmask(image, center, redius)
        mask_ori = cropmask(mask, center, redius)
        tmp_redius = round(redius*enlarge)

        #选择粘贴位置中心点
        new_centers = random_add_patches2(image, liver_mask, mask, redius, newcenters, radiuss, enlarge,  paste_number, iou_thresh=0)
        if len(new_centers) !=0:
            print('new center:', new_centers)
            for new_center in new_centers:
                w_c = new_center[0] - center[0]
                h_c = new_center[1] - center[1]
                for i in range(center[0]-tmp_redius, center[0]+tmp_redius):
                    for j in range(center[1]-tmp_redius, center[1]+tmp_redius):
                        if mask_ori[j][i]!=0:
                            
                            ni = j+h_c 
                            nj = i+w_c
                            newpoint = (nj,ni)
                            if image[newpoint[1]][newpoint[0]]!=0 and image[j][i]!=0:
                                image1[newpoint[1]][newpoint[0]] = image[j][i]
                            elif mask_ori[j][i]!=0:
                                    image1[newpoint[1]][newpoint[0]] = image[j][i]
                
                
                for i in range(center[0]-redius, center[0]+redius):
                    for j in range(center[1]-redius, center[1]+redius):
                        if mask_ori[j][i]!=0:
                            newpoint =[i+w_c, j+h_c]
                            mask1[newpoint[1]][newpoint[0]] = 255
                
                newcenters.append(new_center)
                radiuss.append(tmp_redius)
                
    savename = save_base_dir+'/'+label_dir.split('/')[-1][:-3]+'png'
    print(savename)
    cv2.imwrite(savename, image1)

    savemaskname =  save_mask_dir+'/'+label_dir.split('/')[-1][:-3]+'png'
    print(savemaskname)
    cv2.imwrite(savemaskname, mask1)

#scale
def copysmallobjects2_scale(image_dir, label_dir, mask_dir, save_base_dir, save_mask_dir, enlarge, paste_number):

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
    
    super = 1.5
    for i, center  in enumerate(centers):
        redius = radiuss[i]
        #裁减原始边长1.5倍的区域
        crop_ori = cropmask(image, center, round(super*redius))
        mask_ori = cropmask(mask, center, redius)
        tmp_redius = round(redius*enlarge)

        crop_tmp = crop_ori[center[1]-round(super*redius):center[1]+round(super*redius), center[0]-round(super*redius):center[0]+round(super*redius)]
        cropmask_tmp = mask_ori[center[1]-redius:center[1]+redius, center[0]-redius:center[0]+redius]
        
        #scale
        if np.count_nonzero(mask_ori)<=2048:
            scale_f = 0
            if mask_ori.sum()>1024:
                scale_f = np.random.randint(80,90)/100
            else:
                scale_f = np.random.randint(110,120)/100
            print(scale_f)
            #对裁减区域放大或缩小
            newredius = round(redius*scale_f)
            crop_tmp1 = cv2.resize(crop_tmp, (2*round(newredius*super), 2*round(newredius*super)))
            cropmask_tmp1 = cv2.resize(cropmask_tmp, (2*newredius, 2*newredius))
            
            #resize_crop_mask = cropmask(mask, center, newredius)
            newcrop_tmp = np.zeros_like(crop_ori)
            newcropmask_tmp = np.zeros_like(crop_ori)

            newcrop_tmp[center[1]-round(newredius*super):center[1]+round(newredius*super), center[0]-round(newredius*super):center[0]+round(newredius*super)] = crop_tmp1
            newcropmask_tmp[center[1]-newredius:center[1]+newredius, center[0]-newredius:center[0]+newredius] = cropmask_tmp1
            #if np.abs((newcropmask_tmp*resize_crop_mask).sum()-resize_crop_mask.sum())<5:

            new_centers = random_add_patches2(image, liver_mask, mask, redius, newcenters, radiuss, enlarge,  paste_number=1, iou_thresh=0)
            if len(new_centers) !=0:
                for new_center in new_centers:
                    w_c = new_center[0] - center[0]
                    h_c = new_center[1] - center[1]
                    for i in range(center[0]-newredius,center[0]+newredius):
                        for j in range(center[1]-newredius,center[1]+newredius):
                            if newcropmask_tmp[j][i]!=0:
                                ni = j+h_c 
                                nj = i+w_c
                                newpoint = (nj,ni)
                                if image[newpoint[1]][newpoint[0]]!=0:
                                    image1[newpoint[1]][newpoint[0]] = newcrop_tmp[j][i]
                                elif mask_ori[j][i]!=0:
                                    image1[newpoint[1]][newpoint[0]] = image[j][i]
                

                    for i in range(center[0]-newredius,center[0]+newredius):
                        for j in range(center[1]-newredius,center[1]+newredius):
                            if newcropmask_tmp[j][i]!=0:
                                newpoint =[ i+w_c, j+h_c ]
                                mask1[newpoint[1]][newpoint[0]] = 255

                    newcenters.append(new_center)
                    radiuss.append(newredius)
            

    savename = save_base_dir+'/'+label_dir.split('/')[-1][:-3]+'png'
    print(savename)
    cv2.imwrite(savename, image1)

    savemaskname =  save_mask_dir+'/'+label_dir.split('/')[-1][:-3]+'png'
    print(savemaskname)
    cv2.imwrite(savemaskname, mask1)


#rotate
def fill_img(test_array):
    for i in range(512):
        for j in range(512):
            if test_array[i][j]==0 and test_array[i-1][j]!=0 and  test_array[i][j-1]!=0 and test_array[i+1][j]!=0  and  test_array[i][j+1]!=0:
                newpoint = round((test_array[i-1][j]/4 + test_array[i][j-1]/4+ test_array[i+1][j]/4+test_array[i][j+1]/4))
                test_array[i][j] = newpoint
    return test_array

def fill_mask( im_in):
    im_floodfill = im_in.copy()
    # Mask 用于 floodFill，官方要求长宽+2
    h, w = im_in.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    # floodFill函数中的seedPoint对应像素必须是背景
    isbreak = False
    for i in range(im_floodfill.shape[0]):
        for j in range(im_floodfill.shape[1]):
            if(im_floodfill[i][j]==0):
                seedPoint=(i,j)
                isbreak = True
                break
        if(isbreak):
            break
    
    # 得到im_floodfill 255填充非孔洞值
    cv2.floodFill(im_floodfill, mask,seedPoint, 255)
    # 得到im_floodfill的逆im_floodfill_inv
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    # 把im_in、im_floodfill_inv这两幅图像结合起来得到前景
    im_out = im_in | im_floodfill_inv
    return im_out


def rotate_point(point1, point2, angle, height):
    """
    点point1绕点point2旋转angle后的点
    ======================================
    在平面坐标上，任意点P(x1,y1)，绕一个坐标点Q(x2,y2)旋转θ角度后,新的坐标设为(x, y)的计算公式：
    x= (x1 - x2)*cos(θ) - (y1 - y2)*sin(θ) + x2 ;
    y= (x1 - x2)*sin(θ) + (y1 - y2)*cos(θ) + y2 ;
    ======================================
    将图像坐标(x,y)转换到平面坐标(x`,y`)：
    x`=x
    y`=height-y
    :param point1:
    :param point2: base point (基点)
    :param angle: 旋转角度，正：表示逆时针，负：表示顺时针
    :param height:
    :return:
    """
    x1, y1 = point1
    x2, y2 = point2
    # 将图像坐标转换到平面坐标
    y1 = height - y1
    y2 = height - y2
    x = int(round((x1 - x2) * np.cos(np.pi / 180.0 * angle) - (y1 - y2) * np.sin(np.pi / 180.0 * angle) + x2))
    y = (x1 - x2) * np.sin(np.pi / 180.0 * angle) + (y1 - y2) * np.cos(np.pi / 180.0 * angle) + y2
    # 将平面坐标转换到图像坐标
    y = int(round(height - y))
    return (x, y)

def copysmallobjects2_rotate(image_dir, label_dir, mask_dir, save_base_dir, save_mask_dir, enlarge, paste_number):

    image = cv2.imread(image_dir, 0)
    #image1 = image.copy()
    liver_mask = np.where(image>0, 1, 0).astype(np.uint8)
    mask = cv2.imread(mask_dir, 0)
    mask1 = mask.copy()
    labels = read_label_txt(label_dir)
    if len(labels) == 0:
        return
    centers, radiuss= rescale_yolo_labels(labels, image.shape)  # 转换坐标表示
    newcenters = centers.copy()
   
    for i, center  in enumerate(centers):
        redius = radiuss[i]
        
        crop_ori = cropmask(image, center, redius)
        mask_ori = cropmask(mask, center, redius)
        tmp_redius = enlarge*redius
        
        new_centers = random_add_patches2(image, liver_mask, mask, redius, newcenters, radiuss, enlarge,  paste_number=1, iou_thresh=0)
        if len(new_centers) !=0:
            print('new center:', new_centers)
            for new_center in new_centers:
                w_c = new_center[0] - center[0]
                h_c = new_center[1] - center[1]
                rotate_angle = int(np.random.uniform(-180, 180))
                print('angle', rotate_angle)
                image1 = np.zeros_like(image)
                for i in range(center[0]-tmp_redius, center[0]+tmp_redius):
                    for j in range(center[1]-tmp_redius, center[1]+tmp_redius):
                        if mask_ori[j][i]!=0:
                            
                            ni = j+h_c 
                            nj = i+w_c
                            point = (nj,ni)
                            #对原始粘贴坐标旋转后再粘贴
                            newpoint = rotate_point(point, new_center,rotate_angle, 512)
                            if image[newpoint[1]][newpoint[0]]!=0 and image[j][i]!=0:
                                image1[newpoint[1]][newpoint[0]] = image[j][i]
                            elif mask_ori[j][i]!=0:
                                image1[newpoint[1]][newpoint[0]] = image[j][i]
                #旋转粘贴后的病灶可能有空洞，填补空洞
                image1 = fill_img(image1)
                image1 = np.where(image1>0, image1, image)
                    
                for i in range(center[0]-redius, center[0]+redius):
                    for j in range(center[1]-redius, center[1]+redius):
                        if mask_ori[j][i]!=0:
                            point =[i+w_c, j+h_c]
                            newpoint = rotate_point(point, new_center,rotate_angle, 512)
                            mask1[newpoint[1]][newpoint[0]] = 255
                mask1 = fill_mask(mask1)
                newcenters.append(new_center)
                radiuss.append(tmp_redius)

    savename = save_base_dir+'/'+label_dir.split('/')[-1][:-3]+'png'
    print(savename)
    cv2.imwrite(savename, image1)

    savemaskname =  save_mask_dir+'/'+label_dir.split('/')[-1][:-3]+'png'
    print(savemaskname)
    cv2.imwrite(savemaskname, mask1)
