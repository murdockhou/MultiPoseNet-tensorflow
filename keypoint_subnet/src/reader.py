# encoding: utf-8
'''
@author: shiwei hou
@contact: murdockhou@gmail.com
@software: PyCharm
@file: reader.py
@time: 18-9-28 上午11:53
'''

import tensorflow as tf
import matplotlib.pyplot as plt
import os, json
from src.model import Keypoint_Subnet

class Keypoint_Reader:
    def __init__(self, tfrecord_file, img_size=224, batch_size=4, capacity = 1000, num_threads=12, name=''):
        self.tfrecord_file = tfrecord_file
        self.img_size = img_size
        self.batch_size = batch_size
        self.capacity = capacity
        self.num_threads = num_threads
        self.name = name
        self.reader = tf.TFRecordReader()

    def feed(self):
        with tf.name_scope(self.name):
            filename_queue = tf.train.string_input_producer([self.tfrecord_file])
            _, serialized_example = self.reader.read(filename_queue)
            features = tf.parse_single_example(
                serialized_example,
                features={
                    'image':tf.FixedLenFeature([], tf.string),
                    'id': tf.FixedLenFeature([], tf.string),
                    'height': tf.FixedLenFeature([], tf.int64),
                    'width': tf.FixedLenFeature([], tf.int64)
                })

            img = tf.image.decode_image(features['image'], channels=3) # tensor, [height, width, channels]
            img_id = features['id']
            img_height = tf.cast(features['height'], tf.int32)
            img_width = tf.cast(features['width'], tf.int32)

            img = tf.reshape(img, shape=[img_height, img_width, 3])
            img = self.image_preprocessing(img)

            img_batch, img_id_batch, img_height_batch, img_width_batch = tf.train.shuffle_batch(
                [img, img_id, img_height, img_width],
                batch_size=self.batch_size,
                num_threads=self.num_threads,
                capacity=self.capacity,
                min_after_dequeue=self.capacity // 10
            )

            return img_batch, img_id_batch, img_height_batch, img_width_batch

    def image_preprocessing(self, image):

        img = tf.expand_dims(image, axis=0)
        img = tf.image.resize_nearest_neighbor(img, (self.img_size, self.img_size))
        img = tf.squeeze(img, axis=0)
        return img