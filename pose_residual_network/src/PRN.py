# encoding: utf-8
'''
@author: shiwei hou
@contact: murdockhou@gmail.com
@software: PyCharm
@file: PRN.py
@time: 18-9-27 下午2:36
'''
from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow.contrib.slim import nets
from tensorflow.contrib.layers.python.layers import utils
import tensorflow.contrib.slim as slim

import numpy as np
import os

class PRN(object):
    def __init__(self, inputs, output_node, is_training=True, hidden_node = 1024):
        self.x           = inputs
        self.output_node = output_node
        self.hidden_node = hidden_node
        self.is_training = is_training



    def forward(self):
        with tf.variable_scope('pose-residual-network'):
            flatten  = slim.flatten(inputs=self.x)
            fc1      = slim.fully_connected(inputs=flatten, num_outputs=self.hidden_node, activation_fn=tf.nn.relu)
            dropout1 = slim.dropout(inputs=fc1, is_training=self.is_training)
            fc2      = slim.fully_connected(inputs=dropout1, num_outputs=self.hidden_node, activation_fn=tf.nn.relu)
            dropout2 = slim.dropout(inputs=fc2, is_training=self.is_training)
            fc3      = slim.fully_connected(inputs=dropout2, num_outputs=self.output_node, activation_fn=tf.nn.relu)
            # out      = tf.nn.relu(dropout2)
            out      = tf.add(flatten, fc3)
            out      = tf.nn.softmax(out)
            out      = tf.reshape(out, shape=self.x.get_shape())

            return out
