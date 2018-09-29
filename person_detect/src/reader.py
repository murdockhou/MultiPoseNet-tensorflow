# encoding: utf-8
'''
@author: shiwei hou
@contact: murdockhou@gmail.com
@software: PyCharm
@file: reader.py
@time: 18-9-28 下午2:39
'''

import tensorflow as tf
from src.retinanet import RetinaNet

class Box_Reader(object):
    def __init__(self, tfrecord_file, img_size=224, batch_size=1):
        self.img_size      = img_size
        self.batch_size    = batch_size
        self.tfrecord_file = tfrecord_file
        self.reader        = tf.TFRecordReader()

    def feed(self):
        filename_queue = tf.train.string_input_producer([self.tfrecord_file])
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)

        features = tf.parse_single_example(
            serialized_example,
            features={
                'image': tf.FixedLenFeature((), tf.string),
                'format': tf.FixedLenFeature((), tf.string, 'jpeg'),
                'height': tf.FixedLenFeature([], tf.int64),
                'width': tf.FixedLenFeature([], tf.int64),
                'channel': tf.FixedLenFeature([], tf.int64),
                'boxes': tf.VarLenFeature(dtype=tf.float32),
                'labels': tf.VarLenFeature(dtype=tf.int64)
            }
        )
        channel = tf.cast(features['channel'], tf.int64)
        img = tf.image.decode_jpeg(features['image'], channels=3)  # tensor, [height, width, channels]
        # img = tf.decode_raw(features['image'], tf.uint8)
        img = tf.image.convert_image_dtype(img, dtype=tf.float32)
        img_height = tf.cast(features['height'], tf.int32)
        img_width = tf.cast(features['width'], tf.int32)

        # img = tf.reshape(img, shape=[img_height, img_width, 3])
        # img = (img - 0) / 255  # network image input need to be float type
        # img = tf.to_float(img)

        # features['boxes'] && features['lables'] both SparseTensor type, to get real value stored, need get attribution 'values'
        boxs = features['boxes'].values
        label = features['labels'].values
        # must identify boxs shape and labels shape, otherwise program can not get the shape correctlly
        boxs = tf.reshape(boxs, shape=(30, 4))
        label = tf.reshape(label, shape=(30,))

        if True:
            img, boxs = self._pre_processing(img, img_height, img_width, boxs)

        imgs, heights, widths, boxes, labels = tf.train.batch(
            [img, img_height, img_width, boxs, label],
            batch_size=self.batch_size,
            num_threads=12,
            capacity=1000,
            dynamic_pad=True
        )

        return imgs, heights, widths, boxes, labels

    def _pre_processing(self, img, height, width, bbox):
        img = tf.expand_dims(img, axis=0)
        img = tf.image.resize_nearest_neighbor(img, (self.img_size, self.img_size))
        img = tf.squeeze(img, axis=0)

        factorx = tf.to_float(self.img_size) / tf.to_float(width)
        factory = tf.to_float(self.img_size) / tf.to_float(height)

        bbox = tf.concat([tf.reshape(bbox[:, 0] * factory, (-1, 1)),
                          tf.reshape(bbox[:, 1] * factorx, (-1, 1)),
                          tf.reshape(bbox[:, 2] * factory, (-1, 1)),
                          tf.reshape(bbox[:, 3] * factorx, (-1, 1))],
                         axis=1)
        return img, bbox