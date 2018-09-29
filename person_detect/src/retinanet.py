# encoding: utf-8
'''
@author: shiwei hou
@contact: murdockhou@gmail.com
@software: PyCharm
@file: retinanet.py
@time: 18-9-28 下午2:17
'''

from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow.contrib.slim import nets
import tensorflow.contrib.slim as slim
import math

class RetinaNet(object):
    def __init__(self, fpn, feature_map_dict, batch_size, num_classes, num_anchors=9, is_training=True):
        self.feature_pyramid  = fpn
        self.feature_map_dict = feature_map_dict
        self.batch_size       = batch_size
        self.num_classes      = num_classes
        self.num_anchors      = num_anchors
        self.is_training      = is_training
        self.stddev           = 0.01
        self.pai              = 0.01

    def add_fcn_head(self, inputs, outputs, head_offset):
        with slim.arg_scope([slim.conv2d], scope=str(head_offset), activation_fn=tf.nn.relu,
                            weights_initializer=tf.random_normal_initializer(stddev=self.stddev)):
            net = slim.repeat(inputs, 4, slim.conv2d, 256, kernel_size=[3,3])
            if str(head_offset)[-1] == 's':
                net = slim.conv2d(net, outputs, kernel_size=[3,3], scope=str(head_offset) +'_final', activation_fn=None,
                                  weights_initializer=tf.constant_initializer(0),
                                  biases_initializer=tf.constant_initializer(-(math.log((1-self.pai)/self.pai))))
            else:
                net = slim.conv2d(net, outputs, kernel_size=[3,3], activation_fn=None, scope=str(head_offset) + '_final')

            return net

    def forward(self):
        loc_predictions   = []
        class_predictions = []
        with tf.variable_scope('retina_net'):
            # add P6 and P7 as noticed in papar focal loss, page 4, annotation 2
            self.feature_pyramid['P6'] = slim.conv2d(self.feature_map_dict['C5'], num_outputs=256, kernel_size=[3, 3],
                                                     stride=2,
                                                     weights_initializer=tf.random_normal_initializer(
                                                         stddev=self.stddev),
                                                     activation_fn=None,
                                                     scope='build_fpn_P6')
            self.feature_pyramid['P7'] = slim.conv2d(inputs=(tf.nn.relu(self.feature_pyramid['P6'])),
                                                     num_outputs=256, kernel_size=[3, 3], stride=2,
                                                     weights_initializer=tf.random_normal_initializer(
                                                         stddev=self.stddev),
                                                     activation_fn=None,
                                                     scope='build_fpn_P7')
            # remove P2
            del self.feature_pyramid['P2']

            for idx, feature_map in self.feature_pyramid.items():
                # print ('idx {} crossponding feature map {}'.format(idx, feature_map.get_shape()))
                loc_prediction   = self.add_fcn_head(feature_map, self.num_anchors * 4, str(idx) + '_bbox')
                class_prediction = self.add_fcn_head(feature_map, self.num_classes * self.num_anchors,
                                                     str(idx) + '_class')

                loc_prediction   = tf.reshape(loc_prediction, [self.batch_size, -1, 4])
                class_prediction = tf.reshape(class_prediction, [self.batch_size, -1, self.num_classes])

                loc_predictions.append(loc_prediction)
                class_predictions.append(class_prediction)

            return tf.concat(loc_predictions, axis=1), tf.concat(class_predictions, axis=1)