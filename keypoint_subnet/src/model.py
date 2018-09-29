# encoding: utf-8
'''
@author: shiwei hou
@contact: murdockhou@gmail.com
@software: PyCharm
@file: keypoint_subnet.py
@time: 18-9-28 上午11:23
'''

from __future__ import absolute_import, division, print_function

import tensorflow as tf

import  numpy as np

import os, json
from tensorflow.contrib.slim import nets
from tensorflow.contrib.layers.python.layers import utils
import tensorflow.contrib.slim as slim

from src.backbone import BackBone


class Keypoint_Subnet(object):
    def __init__(self, inputs, img_size, fpn, num_classes, batch_size, is_training=True):
        self.inputs          = inputs
        self.img_size        = img_size
        self.feature_pyramid = fpn
        self.num_classes     = num_classes
        self.batch_size      = batch_size
        self.is_training     = is_training
        self.stddev          = 0.01

        self.input_heats     = tf.placeholder(tf.float32, [self.batch_size, self.img_size // 4, self.img_size // 4, self.num_classes])

        # self.output, self.end_points = self.network()

    def forward(self):
        with tf.variable_scope('keypoint_subnet') as sc:
            end_points_collection = sc.original_name_scope + '_end_points'
            #---------------------------------build layer D--------------------------------#
            feature_d = {}
            for layer in range(2, 6, 1):
                cur_p = self.feature_pyramid['P' + str(layer)]
                d = slim.conv2d(cur_p,
                                num_outputs=128,
                                kernel_size=[3, 3],
                                stride=1,
                                weights_initializer=tf.random_normal_initializer(stddev=self.stddev),
                                scope='build_feature_D%d_1' % layer)
                d = slim.conv2d(d,
                                num_outputs=128,
                                kernel_size=[3, 3],
                                stride=1,
                                weights_initializer=tf.random_normal_initializer(stddev=self.stddev),
                                scope='build_feature_D%d_2' % layer)
                feature_d['D' + str(layer)] = d
            #--------------------------------concat part layer D---------------------------#
            concat_d = feature_d['D2']
            up_shape = concat_d.get_shape()
            up_sample = tf.image.resize_nearest_neighbor(feature_d['D3'], [up_shape[2], up_shape[2]],
                                                         name='upsamping_D3')
            concat_d = tf.concat([concat_d, up_sample], 3)

            up_sample = tf.image.resize_nearest_neighbor(feature_d['D4'], [up_shape[2], up_shape[2]],
                                                         name='upsamping_D4')
            concat_d = tf.concat([concat_d, up_sample], 3)

            up_sample = tf.image.resize_nearest_neighbor(feature_d['D5'], [up_shape[2], up_shape[2]],
                                                         name='upsamping_D5')
            concat_d = tf.concat([concat_d, up_sample], 3)
            #------------------------------via 3x3 conv and relu---------------------------#
            concat_d = slim.conv2d(concat_d,
                                   num_outputs=concat_d.get_shape()[3],
                                   kernel_size=[3, 3],
                                   activation_fn=tf.nn.relu,
                                   weights_initializer=tf.random_normal_initializer(stddev=self.stddev),
                                   scope='smoothed_concat_d_layer')

            #----------------------------------final output--------------------------------#
            output = slim.conv2d(concat_d,
                                 num_outputs=self.num_classes,
                                 kernel_size=[1, 1],
                                 activation_fn=None,
                                 weights_initializer=tf.random_normal_initializer(stddev=self.stddev),
                                 scope='output')

            end_points = utils.convert_collection_to_dict(end_points_collection)

            return output, end_points

    def net_loss(self):
        output, end_points = self.forward()
        out_all            = []
        #-------------------------------add intermediate output loss------------------------------#
        for index, layer in self.feature_pyramid.items():
            layer = tf.image.resize_bicubic(layer, [self.feature_pyramid['P2'].get_shape()[1], self.feature_pyramid['P2'].get_shape()[1]],
                                            name='upsamling_layer_%s' % index)

            output_mid = slim.conv2d(layer, num_outputs=self.num_classes,
                                     kernel_size=[1, 1],
                                     activation_fn=None,
                                     weights_initializer=tf.random_normal_initializer(stddev=self.stddev),
                                     scope='mid_out_%s' % index
                                     )

            out_all.append(output_mid)

        out_all.append(output)
        #---------------------------------------calculate losses----------------------------------#
        losses = []
        for idx, pre_heat in enumerate(out_all):
            loss_l2 = tf.nn.l2_loss(tf.concat(pre_heat, axis=0) - self.input_heats, name='loss_%d' % idx)
            losses.append(loss_l2)

        total_loss = tf.reduce_sum(losses) / self.batch_size
        net_out_loss = tf.reduce_sum(loss_l2) / self.batch_size
        #-----------------------------------------add tf summary----------------------------------#
        tf.summary.scalar('total_loss', total_loss)
        tf.summary.scalar('net_loss', net_out_loss)
        tf.summary.image('ori_image', self.inputs, max_outputs=2)
        tf.summary.image('gt_head', tf.reshape(tf.transpose(
            self.input_heats, [3, 0, 1, 2])[12],shape=[-1, self.img_size // 4, self.img_size // 4, 1]), max_outputs=2)
        tf.summary.image('pred_head', tf.reshape(tf.transpose(
            pre_heat, [3, 0, 1, 2])[12], shape=[-1, self.img_size // 4, self.img_size // 4, 1]),max_outputs=2)

        return total_loss, net_out_loss, pre_heat

if __name__ == '__main__':
    graph = tf.Graph()
    with graph.as_default():
        batch_size = 1
        height, width = 224, 224
        inputs = tf.random_uniform((batch_size, height, width, 3), seed=1)

        backbone = BackBone(img_size = 224, batch_size=1)
        fpn, _ = backbone.build_fpn_feature()
        kp = Keypoint_Subnet(backbone.input_imgs, img_size=backbone.img_size, fpn=fpn, batch_size=backbone.batch_size, num_classes=14)
        total_loss, net_loss, pre_heat = kp.net_loss()
        init = tf.group(
            tf.global_variables_initializer(),
            tf.local_variables_initializer())

        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(init)

            writer = tf.summary.FileWriter('graph', tf.get_default_graph())
            writer.close()
