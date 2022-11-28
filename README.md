# LSM
liver lesions augmentation

demo_hq3.py参数说明
  enlarge:控制外扩倍数，如果矩形即为边长外扩倍数，如果圆为半径扩大倍数
  paste_number:粘贴次数
  
1、aug3_oricp.py
  函数copysmallobjects2_oricp()对原始病灶复制粘贴
  函数copysmallobjects2_scale()对病灶缩放后粘贴，为避免放大后病灶边界有过渡色，通过super参数裁减比病灶本身大1.5倍的区域,对该区域缩放后再粘贴
  函数copysmallobjects2_rotate()对病灶旋转后复制粘贴，复制区域逐像素旋转后再粘贴到新位置,旋转后的区域可能出现空洞,fill_img()对i可能的空洞填补
2、病灶最小外接矩形粘贴aug3_rectangle.py
  copysmallobjects2_rectangle()
3、病灶最小外接圆粘贴aug3_oricp.py
  copysmallobjects2_circle()
4、病灶polygon粘贴，主程序demo_hq3_ply.py，参数enlarge轮廓外扩像素，dtype为context信息融合方式,

                 融合方式    代码
    no          不融合      aug3_dilate.py
    ave         加权平均    aug3_dilate.py
    gauss       高斯       aug3_dilate_gauss.py
    reverse     反向高斯    aug3_dilate_gauss.py
