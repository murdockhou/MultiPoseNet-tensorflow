# encoding: utf-8
'''
@author: shiwei hou
@contact: murdockhou@gmail.com
@software: PyCharm
@file: get_heatmap.py
@time: 18-8-28 下午4:50
'''

import numpy as np
import math, cv2
from skimage.filters import gaussian

def get_heatmap(label_dict, img_ids, img_heights, img_widths, img_resize, num_keypoints, sigma = 6.0):
    batch = img_ids.shape[0]
    heatmaps = np.zeros([batch, img_resize//4, img_resize//4, num_keypoints], np.float32)

    for b in range(batch):
        height = img_heights[b]
        width = img_widths[b]
        keypoints = label_dict[img_ids[b].decode('utf-8')]

        single_heatmap = get_single_heatmap(keypoints, height, width, num_keypoints, sigma)
        single_heatmap = cv2.resize(single_heatmap, (img_resize//4, img_resize//4))

        heatmaps[b,:,:,:] = single_heatmap

    return heatmaps

def get_single_heatmap(keypoints, height, width, channels, sigma = 6.0):
    heatmap = np.zeros([channels, height, width], np.float32)
    keypoints = list(keypoints)
    keypoints = np.asarray(keypoints)
    keypoints = np.reshape(keypoints, (len(keypoints)//channels//3, channels*3))

    for people in keypoints:
        for i in range (channels):
            keypoint_x = people[i*3]
            keypoint_y = people[i*3+1]
            keypoint_v = people[i*3+2]

            if keypoint_x == 0 and keypoint_y == 0:
                continue
            if keypoint_v == 3:
                continue

            heatmap = put_keypoint_on_heatmap(keypoint_x, keypoint_y, i, heatmap, sigma)
            # heatmap[i, keypoint_y, keypoint_x] = 1

    # heatmap = gaussian(heatmap.transpose((1, 2, 0)), sigma=sigma, mode='constant', multichannel=True)
    return heatmap.transpose((1, 2, 0))

def put_keypoint_on_heatmap(center_x, center_y, channel, heatmap, sigma = 6.0):
    th = 1.6052
    delta = math.sqrt(th * 2)

    height = heatmap.shape[1]
    width = heatmap.shape[2]

    x0 = int(max(0, center_x - delta * sigma))
    y0 = int(max(0, center_y - delta * sigma))

    x1 = int(min(width, center_x + delta * sigma))
    y1 = int(min(height, center_y + delta * sigma))

    for y in range(y0, y1):
        for x in range(x0, x1):
            d = (x - center_x) ** 2 + (y - center_y) ** 2
            exp = d / 2.0 / sigma / sigma

            if exp > th:
                continue
            heatmap[channel][y][x] = max(heatmap[channel][y][x], math.exp(-exp))
            heatmap[channel][y][x] = min(heatmap[channel][y][x], 1.0)

    return heatmap