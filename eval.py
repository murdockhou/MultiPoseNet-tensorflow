# encoding: utf-8
'''
@author: shiwei hou
@contact: murdockhou@gmail.com
@software: PyCharm
@file: eval.py
@time: 18-9-29 下午6:56
'''

import tensorflow as tf
from keypoint_subnet.src.backbone import BackBone
from keypoint_subnet.src.model import Keypoint_Subnet
from person_detect.src.retinanet import RetinaNet
from pose_residual_network.src.PRN import PRN

import numpy as np
import cv2, os, json, time
from datetime import datetime

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer('img_size', 480, '')
tf.flags.DEFINE_integer('batch_size', 1, '')
tf.flags.DEFINE_integer('num_keypoints', 17, '')
tf.flags.DEFINE_integer('num_classes', 1, '')
tf.flags.DEFINE_string('keypoint_trained_model', '', 'the checkpoint path of keypoint_subnet trained model.')
tf.flags.DEFINE_string('person_detect_trained_model', '', 'the checkpoint path of person_detect trained model.')
tf.flags.DEFINE_string('PRN trained model', '', 'the checkpoint path of PRN trained model.')

def eval(inputs):
    tf.reset_default_graph()
    graph = tf.Graph()
    with graph.as_default():
        #----------------------------------net work-----------------------------------------#
        #--------------------------------1. backbone----------------------------------------#
        backbone              = BackBone(img_size=FLAGS.img_size, batch_size=FLAGS.batch_size, is_training=False)
        fpn, feature_map_dict = backbone.build_fpn_feature()
        #--------------------------------2. keypoint net-------------------------------------#
        keypoint_net    = Keypoint_Subnet(inputs=backbone.input_imgs, img_size=FLAGS.img_size, fpn=fpn, num_classes=FLAGS.num_keypoints,
                                          batch_size=FLAGS.batch_size, is_training=False)
        pred_heatmap, _ = keypoint_net.forward()
        #--------------------------------3. person detect------------------------------------#
        retina_net         = RetinaNet(fpn=fpn, feature_map_dict=feature_map_dict, batch_size=FLAGS.batch_size,
                                       num_classes=FLAGS.num_classes+1, is_training=False)
        loc_pred, cls_pred = retina_net.forward()
        #--------------------------------4. pose-residual-net--------------------------------#
        # TODO 1. deal pred_heatmap, resize it based on img_size
        # TODO 2. select pred_box based on (loc_pred, cls_pred)
        # TODO 3. combine result 1 and 2, make it to prn network input format
        # TODO 4. deal with the output of 3, and put it on ori_img
