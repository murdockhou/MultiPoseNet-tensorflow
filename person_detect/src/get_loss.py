# encoding: utf-8
'''
@author: shiwei hou
@contact: murdockhou@gmail.com
@software: PyCharm
@file: get_loss.py
@time: 18-9-28 下午2:53
'''

import tensorflow as tf
import numpy as np

from anchor.anchor_generator import create_retinanet_anchors, anchor_assign
from anchor.box_coder import FasterRCNNBoxCoder
from anchor.box_list import BoxList

from src.loss import focal_loss, regression_loss

def get_loss(img_size, batch_size, gt_boxes, loc_pred, gt_labels, cls_pred, num_classes=1, is_training=True):

    #--------------------based gt get anchors_list------------------------#
    anchors_list = get_inputs(img_size=img_size, batch_size=batch_size, gt_boxes=gt_boxes,
                              gt_labels=gt_labels, is_training=is_training)
    #-----------------------------net-------------------------------------#
    # backbone = BackBone(img_size, batch_size, is_training=is_training)
    # fpn      = backbone.build_fpn_feature()
    # net      = RetinaNet(fpn=fpn, batch_size=batch_size, num_classes=num_classes, is_training=is_training)
    # loc_pred, cls_pred = net.forward()

    # ----------------------decode pred_boxs-----------------------#
    # ----convert [ty, tx, th, tw] to [ymin, xmin, ymax, xmax]-----#
    decoded_loc_pred = []
    for i in range(batch_size):
        anchor = anchors_list[i]
        current_loc_pred = loc_pred[i]
        # 根据anchor将网络的loc输出解码，表示为[ymin, xmin, ymax, xmax]
        current_box_list = FasterRCNNBoxCoder().decode(current_loc_pred, anchor.get())
        current_decoded_loc_pred = current_box_list.get()
        decoded_loc_pred.append(current_decoded_loc_pred)

    #---------get num of anchor overlapped with ground truth box------------#
    cls_gt = [anchor.get_field('gt_labels') for anchor in
              anchors_list]  # a list, contains batchs number tensor, each tensor is 1D contains #anchors label
    loc_gt = [anchor.get_field('gt_encoded_boxes') for anchor in
              anchors_list]  # a list, contains batchs number tensor, each tensor (gt_encoded_boxes) shape is [-1, 4],
                             # the format of gt_encoded_boxes is [ymin, xmin, ymax, xmax]
    #--------------------------calculate loss-------------------------------#
    total_loss = 0
    for i in range(batch_size):
        single_cls_gt   = cls_gt[i]  # [#anchors,]
        single_loc_gt   = loc_gt[i]  # [#anchors,4]
        single_cls_pred = cls_pred[i]  # [#anchors,2]
        single_loc_pred = loc_pred[i]  # [#anchors,4]

        # print(single_cls_pred.get_shape(), single_cls_gt.get_shape())

        # focal loss, remove anchor which label equal to -1
        # 因为前面生成的gt_labels，会有的anchor在iou [0.4,0.5)之间，标签为-1，要忽略掉，所以要先把这些去掉
        valid_anchor_indices = tf.where(tf.greater_equal(single_cls_gt, 0))
        valid_cls_gt         = tf.gather_nd(single_cls_gt, valid_anchor_indices)
        valid_cls_pred       = tf.gather_nd(single_cls_pred, valid_anchor_indices)
        cls_gt_onehot        = tf.one_hot(valid_cls_gt, depth=num_classes + 1)  # [#anchors, depth]
        floss                = focal_loss(cls_gt_onehot, valid_cls_pred)

        # location regression loss, remove background which label == 0
        valid_cls_indices = tf.where(tf.greater(single_cls_gt, 0))
        valid_loc_gt      = tf.reshape(tf.gather_nd(single_loc_gt, valid_cls_indices), shape=(-1, 4))
        valid_loc_preds   = tf.reshape(tf.gather_nd(single_loc_pred, valid_cls_indices), shape=(-1, 4))
        loc_loss          = regression_loss(valid_loc_preds, valid_loc_gt)


        total_loss = total_loss + tf.reduce_sum(floss) + tf.reduce_sum(loc_loss)

    loss = tf.to_float(total_loss) / tf.to_float(batch_size)
    return loss, decoded_loc_pred


def get_inputs(img_size, batch_size, gt_boxes, gt_labels, is_training=True):
    loc_gt = gt_boxes
    cls_gt = gt_labels #[batch_size, #gt_anchors_number]

    # print (loc_gt.get_shape(), cls_gt.get_shape())
    # get anchors
    anchors_list = []
    for i in range(batch_size):
        input_size = [tf.to_float(img_size), tf.to_float(img_size)]
        feature_map_list = [(tf.ceil(tf.multiply(input_size[0], 1/pow(2., i+3))),
                             tf.ceil(tf.multiply(input_size[1], 1/pow(2., i+3))))
                            for i in range(5)]
        anchor_generator = create_retinanet_anchors()
        anchor = anchor_generator.generate(input_size, feature_map_list)

        current_loc_gt = loc_gt[i] #[#gt_anchors_number, 4]
        current_cls_gt = cls_gt[i] #[#gt_anchors_number]
        print('Before remove zeros boxs, loc_gt shape = {}, cls_gt shape = {}'.format(current_loc_gt.get_shape(), current_cls_gt.get_shape()))
        current_loc_gt, current_cls_gt = deal_zeros_box(current_loc_gt, current_cls_gt)
        print('After remove zeros boxs, loc_gt shape = {}, cls_gt shape = {}'.format(current_loc_gt.get_shape(), current_cls_gt.get_shape()))


        anchor = anchor_assign(anchor, gt_boxes=BoxList(current_loc_gt), gt_labels=current_cls_gt, is_training=is_training)

        # encode anchor boxes
        gt_boxes = anchor.get_field('gt_boxes')

        encoded_gt_boxes = FasterRCNNBoxCoder().encode(gt_boxes, anchor.get())
        anchor.add_field('gt_encoded_boxes', encoded_gt_boxes)
        anchors_list.append(anchor)

    return anchors_list

def deal_zeros_box(gt_boxes, gt_labels):
    '''
    can not do anything, because one dim in gt_boxes and gt_labels is ?
    update: now, we set ? = 30 in tfrecord file, so we can deal with zeros boxs
    :param gt_boxes: [#boxs, 4]
    :param gt_labels: [#boxs]
    :return:
    '''
    #------------------------deal boxs--------------------------------------------------------#
    gt_boxs = tf.unstack(gt_boxes, axis=0) # gt_boxs, a list contains nums boxs which has shape(4,)
    gt_box = tf.expand_dims(gt_boxes[0], axis=0)
    is_first = True # the first box is always non-zero box

    for box in gt_boxs:
        if is_first:
            is_first = False
            continue
        gt_box = tf.cond(tf.equal(tf.reduce_sum(box), tf.reduce_sum(tf.zeros_like(box))), lambda: gt_box,
                         lambda: tf.concat([gt_box, tf.expand_dims(box, axis=0)], axis=0))


    #---------------------------deal labels--------------------------------------------------#
    gt_labels = tf.unstack(gt_labels, axis=0)
    gt_label = tf.expand_dims(gt_labels[0], axis=0)
    is_first = True # the first label is always non-background
    for label in gt_labels:
        if is_first:
            is_first = False
            continue
        gt_label = tf.cond(tf.equal(tf.reduce_sum(label), tf.reduce_sum(tf.zeros_like(label))), lambda: gt_label,
                           lambda : tf.concat([gt_label, tf.expand_dims(label, axis=0)], axis=0))

    gt_label = tf.reshape(gt_label, shape=(-1,))

    return gt_box, gt_label