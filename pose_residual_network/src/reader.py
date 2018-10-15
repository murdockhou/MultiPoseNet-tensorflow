# encoding: utf-8
'''
@author: shiwei hou
@contact: murdockhou@gmail.com
@software: PyCharm
@file: reader.py
@time: 18-9-27 下午5:05
'''

import tensorflow as tf
import numpy as np
import cv2
import os

class PRN_READER(object):
    def __init__(self, batch_size, height, width, channels, tfrecord_file):
        self.batch_size = batch_size
        self.height        = height
        self.width         = width
        self.channles      = channels
        self.reader        = tf.TFRecordReader()
        self.tfrecord_file = tfrecord_file

    def feed(self):

        filename_queue = tf.train.string_input_producer([self.tfrecord_file], num_epochs=16)
        reader         = self.reader
        _, serialized_example = reader.read(filename_queue)

        features = tf.parse_single_example(
            serialized_example,
            features={
                'input': tf.VarLenFeature(dtype=tf.float32),
                'label': tf.VarLenFeature(dtype=tf.float32)
            }
        )

        inputs = features['input'].values
        label  = features['label'].values

        inputs = tf.reshape(inputs, shape=(self.height, self.width, self.channles))
        label  = tf.reshape(label,  shape=(self.height, self.width, self.channles))

        batch_input, batch_label = tf.train.shuffle_batch(
            [inputs, label],
            batch_size=self.batch_size,
            num_threads=4,
            capacity=1000,
            min_after_dequeue=100
        )

        return batch_input, batch_label


def reader_test():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    reader = PRN_READER(batch_size=1, height=56, width=36, channels=17,
                        tfrecord_file='/raid5/hswData/pose_residual_net_tfrecord/coco_train2017_6.tfrecord')
    net_x, label = reader.feed()
    # net_x = tf.reduce_sum(net_x, axis=3, keepdims=True)
    # label = tf.reduce_sum(label, axis=3, keepdims=True)

    with tf.Session() as sess:
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        coord   = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        step = 0
        try:
            while not coord.should_stop():
                _1, _2 = sess.run([net_x, label])
                step += 1
        except tf.errors.OutOfRangeError:
            print('done. total step == ', step)
        finally:

            print ('batch = 1, epochs = 1,  total step == ', step)
            coord.request_stop()
            coord.join(threads)


if __name__ == '__main__':
    reader_test()