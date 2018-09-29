# encoding: utf-8
'''
@author: shiwei hou
@contact: murdockhou@gmail.com
@software: PyCharm
@file: backbone.py
@time: 18-9-28 上午11:03
'''

from __future__ import absolute_import, division, print_function

import tensorflow as tf

from tensorflow.contrib.slim import nets
from tensorflow.contrib.layers.python.layers import utils
import tensorflow.contrib.slim as slim

class BackBone(object):
    def __init__(self, img_size, batch_size, is_training=True):
        self.img_size    = img_size
        self.batch_size  = batch_size
        self.input_imgs  = tf.placeholder(tf.float32, [self.batch_size, self.img_size, self.img_size, 3])
        self.is_training = is_training
        self.stddev      = 0.01

    def get_feature_map(self):
        #-------------------resent---------------------#
        arg_scope = nets.resnet_v2.resnet_arg_scope()
        with slim. arg_scope(arg_scope):
            out, end_points = nets.resnet_v2.resnet_v2_50(inputs=self.input_imgs, num_classes=None, is_training=self.is_training)
        #---------------feature map dict---------------#
        feature_map_dict = {
            'C2': end_points['resnet_v2_50/block1/unit_2/bottleneck_v2'],  # input_size / 4
            'C3': end_points['resnet_v2_50/block2/unit_3/bottleneck_v2'],  # input_size / 8
            'C4': end_points['resnet_v2_50/block3/unit_4/bottleneck_v2'],  # input_size / 16
            'C5': end_points['resnet_v2_50/block4']                        # input_size / 32
        }
        return feature_map_dict

    def build_fpn_feature(self):
        feature_pyramid  = {}
        feature_map_dict = self.get_feature_map()
        #------------------------------------------build fpn-------------------------------------------#
        with tf.variable_scope('build_fpn_feature'):
            with slim.arg_scope([slim.conv2d], weights_initializer=tf.random_normal_initializer(stddev=self.stddev)):
                feature_pyramid['P5'] = slim.conv2d(feature_map_dict['C5'], num_outputs=256, kernel_size=[1, 1], stride=1,
                                        scope='build_fpn_P5')

                #------------------ top-down pathway and lateral connections--------------------------#
                for layer in range(4, 1, -1):
                    p = feature_pyramid['P' + str(layer + 1)]
                    c = feature_map_dict['C' + str(layer)]

                    #---------------------------------- upsample p -----------------------------------#
                    up_shape = c.get_shape()
                    up_sample = tf.image.resize_nearest_neighbor(p, [up_shape[2], up_shape[2]],
                                                                 name='upsampling_fpn_P%d' % layer)

                    #----------------------------------- 1x1 conv ------------------------------------#
                    c = slim.conv2d(c, num_outputs=256, kernel_size=[1, 1], stride=1, scope='fpn_1x1conv_C%d' % layer)
                    p = up_sample + c

                    #----------------------reduce aliasing effect of upsampling ----------------------#
                    #---------------(in the third last paragraph, Section 3, Paper FPN)---------------#
                    p = slim.conv2d(p, num_outputs=256, kernel_size=[3, 3], stride=1, padding='SAME',
                                    scope='build_fpn_P%d' % layer)

                    feature_pyramid['P' + str(layer)] = p

        return feature_pyramid, feature_map_dict

