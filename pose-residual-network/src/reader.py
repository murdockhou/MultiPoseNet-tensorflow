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

class PRN_READER(object):
    def __init__(self, batch_size, height, width, channels, tfrecord_file):
        self.batch_size = batch_size
        self.height        = height
        self.width         = width
        self.channles      = channels
        self.reader        = tf.TFRecordReader()
        self.tfrecord_file = tfrecord_file

    def feed(self):

        filename_queue = tf.train.string_input_producer([self.tfrecord_file])
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
            num_threads=12,
            capacity=1000,
            min_after_dequeue=100
        )

        return batch_input, batch_label


def reader_test():
    reader = PRN_READER(batch_size=2, height=56, width=36, channels=14,
                        tfrecord_file='/media/ulsee/E/pose_residual_net_tfrecord/ai_test.tfrecord')
    net_x, label = reader.feed()
    # net_x = tf.reduce_sum(net_x, axis=3, keepdims=True)
    # label = tf.reduce_sum(label, axis=3, keepdims=True)

    with tf.Session() as sess:
        coord   = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        img1, img2 = sess.run([net_x, label])
        img1 = np.sum(img1, axis=3, keepdims=True)
        img2 = np.sum(img2, axis=3, keepdims=True)
        # img1 = img1[:,:,4]
        # img2 = img2[:,:,4]
        cv2.imwrite('/media/ulsee/E/pose_residual_net_tfrecord/input2.jpg', img1[1]*255)
        cv2.imwrite('/media/ulsee/E/pose_residual_net_tfrecord/label2.jpg', img2[1]*255)

        print (sess.run(net_x).shape)
        print (sess.run(label).shape)

        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    reader_test()