# encoding: utf-8
'''
@author: shiwei hou
@contact: murdockhou@gmail.com
@software: PyCharm
@file: image_preprocessing.py
@time: 18-8-30 上午11:29
'''

import numpy as np
import os, cv2, random
import matplotlib.pyplot as plt

import sys
sys.path.append('../')
from src.get_heatmap import get_single_heatmap


def img_pre_processing(imgs, heatmaps):
    '''

    :param imgs: image batch, shape = [b, h, w, c]
    :param heatmaps: heatmap batch, shape = [b, h, w, c]
    :return: depend on rd, rotate 45 degree or not, vertically flip or not
    return processing imgs and heatmaps with ori shape
    '''

    batch = imgs.shape[0]
    for i in range(batch):
        current_img = imgs[i, :, :, :]
        current_heatmap = heatmaps[i, :, :, :]

        rd = random.randint(1, 10)
        if rd < 4:
            current_img, current_heatmap = image_rotation(current_img, current_heatmap, 40)

        elif rd > 7:
            current_img, current_heatmap = image_rotation(current_img, current_heatmap, -40)

        rd = random.randint(1, 10)
        if rd < 4:
            current_img, current_heatmap = image_vertical_flipping(current_img, current_heatmap)

        imgs[i,:,:,:] = current_img
        heatmaps[i,:, :, :] = current_heatmap

    return imgs, heatmaps

def image_rotation(img, heatmap, degree=40):
    img_ori_shape = img.shape  # [h, w, c]
    heat_ori_shape = heatmap.shape  # [ h, w, c]

    img = rotated_bound(img, degree)
    img = cv2.resize(img, (img_ori_shape[1], img_ori_shape[0]))

    for c in range(heat_ori_shape[2]):
        cur_heatmap = heatmap[:, :, c]
        cur_heatmap = np.expand_dims(cur_heatmap, axis=2)
        cur_heatmap = rotated_bound(cur_heatmap, degree)
        if len(cur_heatmap.shape) == 3:
            cur_heatmap = np.squeeze(cur_heatmap, axis=2)
        heatmap[:, :, c] = cv2.resize(cur_heatmap, (heat_ori_shape[1], heat_ori_shape[0]))
    return img, heatmap

def rotated_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))

def image_vertical_flipping(img, heatmap):
    '''
    要注意的是，进行flip之后，heatmap各个channel的值要改变，因为flip之后，图片上原本左关节点会变成右关节点，
    右关节点同样会变成左关节点，因此需要在对heatmap进行flip之后，交换左右两个通道。
    coco数据集标注的顺序是：
    [0------16]:
    0:    nose
    1-2:  left eye,      right eye
    3-4:  left ear,      right ear
    5-6:  left shoulder, right shoulder
    7-8:  left elbow,  right elbow
    9-10: left wrist , right wrist
    11-12:left hip,    right hip
    13-14:left knee,   right knee
    15-16:left ankle,  right ankle
    :param img:
    :param heatmap:
    :return:
    '''

    img = cv2.flip(img, 0)
    for i in range(heatmap.shape[2]):
        cur_heat = heatmap[:, :, i]

        cur_heat = np.expand_dims(cur_heat, axis=2)
        cur_heat = cv2.flip(cur_heat, 0)
        if len(cur_heat.shape) == 3:
            cur_heat = np.squeeze(cur_heat, axis=2)

        heatmap[:, :, i] = cur_heat

    # exchane left & right joints
    new_heatmap = np.zeros(heatmap.shape, dtype=heatmap.dtype)
    for i in range(1, 16, 2):
        new_heatmap[:, :, i+1]   = heatmap[:, :, i]
        new_heatmap[:, :, i] = heatmap[:, :, i+1]
    new_heatmap[:, :, 0] = heatmap[:, :, 0]
    return img, new_heatmap

def _test():
    img = cv2.imread('/media/ulsee/E/datasets/coco/cocoval2017/000000281929.jpg', cv2.COLOR_BGR2RGB)
    img_copy = img.copy()
    # cv2.imwrite('gt_img.jpg', img)
    # img = cv2.flip(img, 0)
    kp = [339,93,2,346,88,2,328,88,2,360,89,2,318,90,1,385,135,2,301,147,2,416,184,2,
          286,204,2,407,226,2,276,244,2,358,254,2,309,259,2,352,346,2,307,349,2,348,448,2,312,449,2]
    heatmap = get_single_heatmap(kp, img.shape[0], img.shape[1], channels=17, sigma=4)

    img, heatmap = image_rotation(img, heatmap, 40)
    img, heatmap = image_vertical_flipping(img, heatmap)
    cv2.imwrite('img_rotate_flip.jpg', img)
    #---------#
    for c in range(17):
        ch = heatmap[:, :, c]
        # print (ch)
        curmax = np.max(ch)
        index = np.where(ch == curmax)
        coorx = index[0][0]
        coory = index[1][0]
        cv2.circle(img, (coory, coorx), 5, (0, 0, 255), -1)
        cv2.putText(img, str(c), (coory, coorx), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 1)
    cv2.imwrite('img_rotate_flip_with_heat.jpg', img)
    heatmap = np.sum(heatmap, axis=2, keepdims=True) * 255
    cv2.imwrite('heat_rotate_flip.jpg', heatmap)

    # heatmap_ori = heatmap
    # heatmap_ori = np.sum(heatmap_ori, axis=2, keepdims=True)*255
    # cv2.imwrite('gt_heat.jpg', heatmap_ori)
    # # ---------#
    # for c in range(17):
    #     ch = heatmap[:, :, c]
    #     # print (ch)
    #     curmax = np.max(ch)
    #     index = np.where(ch == curmax)
    #     coorx = index[0][0]
    #     coory = index[1][0]
    #     cv2.circle(img_copy, (coory, coorx), 5, (0, 0, 255), -1)
    #     cv2.putText(img_copy, str(c), (coory, coorx), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 1)
    # cv2.imwrite('img_with_heat.jpg', img_copy)
    # #-----------#
    # img, heatmap = image_vertical_flipping(img, heatmap)
    # # img, heatmap = image_rotation(img, heatmap)
    #
    # cv2.imwrite('img_flip.jpg', img)
    #
    # #---------#
    # for c in range(17):
    #     ch = heatmap[:, :, c]
    #     # print (ch)
    #     curmax = np.max(ch)
    #     index = np.where(ch == curmax)
    #     coorx = index[0][0]
    #     coory = index[1][0]
    #     cv2.circle(img, (coory, coorx), 5, (0, 0, 255), -1)
    #     cv2.putText(img, str(c), (coory, coorx), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 1)
    # cv2.imwrite('img_flip_with_heat.jpg', img)
    # #---------#
    # heatmap = np.sum(heatmap, axis=2, keepdims=True) * 255
    #
    # heatmap = cv2.cvtColor(heatmap, cv2.COLOR_GRAY2RGB)
    # cv2.imwrite('heat_flip.jpg', heatmap)




if __name__ == '__main__':
    _test()
