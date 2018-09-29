# encoding: utf-8
'''
@author: shiwei hou
@contact: murdockhou@gmail.com
@software: PyCharm
@file: draw_box_with_image.py
@time: 18-9-28 下午3:07
'''

import tensorflow as tf
import numpy as np
from src.get_loss import deal_zeros_box

def get_gt_boxs_with_img(imgs, gt_boxs, gt_labels, batch_size, img_size):
    gt_img_batch_with_box = []
    for i in range(batch_size):

        # remove zeros box [0,0,0,0]
        current_loc = gt_boxs[i]
        current_cls = gt_labels[i]
        current_loc, current_cls = deal_zeros_box(current_loc, current_cls)
        current_gt_box           = current_loc / tf.to_float(img_size)

        # draw box on single image
        img_batch_i        = imgs[i]
        output_box_batch_i = tf.expand_dims(current_gt_box, axis=0)
        img_batch_i        = tf.expand_dims(img_batch_i, axis=0)

        img_batch_i_with_box = tf.image.draw_bounding_boxes(images=img_batch_i, boxes=output_box_batch_i)
        gt_img_batch_with_box.append(img_batch_i_with_box)

    gt_img_batch_with_box = tf.reshape(tf.concat(gt_img_batch_with_box, axis=0),
                                       shape=(batch_size, img_size, img_size, 3))
    return gt_img_batch_with_box

def get_pred_boxs_with_img(imgs, decoded_boxs, cls_pred,  batch_size, img_size):

    batch_output_box = []
    batch_output_box_score = []
    for i in range(batch_size):
        box_score = tf.nn.softmax(cls_pred[i])
        box_score = box_score[:, 1]
        top_k_score, top_k_indices = tf.nn.top_k(box_score, k=60)
        decode_boxes  = tf.gather(decoded_boxs[i], top_k_indices)
        valid_indices = tf.image.non_max_suppression(boxes=decode_boxes, scores=top_k_score, max_output_size=6,
                                                     iou_threshold=0.5)
        output_loc   = tf.gather(decode_boxes, valid_indices)
        output_score = tf.gather(top_k_score, valid_indices)
        batch_output_box.append(output_loc)
        batch_output_box_score.append(output_score)

    pred_img_batch_with_box = []
    for i in range(batch_size):
        output_box_batch_i   = batch_output_box[i] / tf.to_float(img_size)
        img_batch_i          = imgs[i]
        output_box_batch_i   = tf.expand_dims(output_box_batch_i, axis=0)
        img_batch_i          = tf.expand_dims(img_batch_i, axis=0)
        img_batch_i_with_box = tf.image.draw_bounding_boxes(images=img_batch_i, boxes=output_box_batch_i)
        pred_img_batch_with_box.append(img_batch_i_with_box)

    pred_img_batch_with_box  = tf.reshape(tf.concat(pred_img_batch_with_box, axis=0),
                                         shape=(batch_size, img_size, img_size, 3))
    return pred_img_batch_with_box
