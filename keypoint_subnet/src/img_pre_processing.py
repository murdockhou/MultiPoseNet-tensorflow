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
            current_img, current_heatmap = image_rotation(current_img, current_heatmap, 45)

        if rd > 7:
            current_img, current_heatmap = image_rotation(current_img, current_heatmap, -45)

        rd = random.randint(1, 10)
        if rd < 6:
            current_img, current_heatmap = image_vertical_flipping(current_img, current_heatmap)

        imgs[i,:,:,:] = current_img
        heatmaps[i,:, :, :] = current_heatmap

    return imgs, heatmaps

def image_rotation(img, heatmap, degree):
    img_ori_shape = img.shape  # [h, w, c]
    heat_ori_shape = heatmap.shape  # [ h, w, c]

    img = rotated_bound(img, degree)
    img = cv2.resize(img, (img_ori_shape[0], img_ori_shape[1]))

    for c in range(heat_ori_shape[2]):
        cur_heatmap = heatmap[:, :, c]
        cur_heatmap = np.expand_dims(cur_heatmap, axis=2)
        cur_heatmap = rotated_bound(cur_heatmap, degree)
        if len(cur_heatmap.shape) == 3:
            cur_heatmap = np.squeeze(cur_heatmap, axis=2)
        heatmap[:, :, c] = cv2.resize(cur_heatmap, (heat_ori_shape[0], heat_ori_shape[1]))
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

    img = cv2.flip(img, 0)
    for i in range(heatmap.shape[2]):
        cur_heat = heatmap[:, :, i]

        cur_heat = np.expand_dims(cur_heat, axis=2)
        cur_heat = cv2.flip(cur_heat, 0)
        if len(cur_heat.shape) == 3:
            cur_heat = np.squeeze(cur_heat, axis=2)

        heatmap[:, :, i] = cur_heat

    return img, heatmap

def _test():
    img = cv2.imread('test.jpg', cv2.COLOR_BGR2RGB)
    kp = [174,63,1,253,154,1,350,183,1,411,354,1,0,0,3,0,0,3,0,0,3,0,0,3,0,0,3,0,0,3]
    heatmap = get_single_heatmap(kp, img.shape[0], img.shape[1], 10, 4)
    img = cv2.resize(img, (224, 224))
    heatmap = cv2.resize(heatmap, (56, 56))

    # img, heatmap = image_rotation(img, heatmap)
    # cv2.imwrite('gt_flip.jpg', img)
    heatmap = np.sum(heatmap, axis=2, keepdims=True) * 255
    # heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_GRAY2RGB)
    cv2.imwrite('heat_.jpg', heatmap)




if __name__ == '__main__':
    _test()
