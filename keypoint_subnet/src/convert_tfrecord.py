# encoding: utf-8
'''
@author: shiwei hou
@contact: murdockhou@gmail.com
@software: PyCharm
@file: convert_tfrecord.py.py
@time: 18-9-28 下午6:50
''' 

import os, cv2
import numpy as np
import tensorflow as tf

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string(name='image_dir', default='/media/ulsee/E/datasets/test2',
                       help='image directory for building tfrecord')

tf.flags.DEFINE_string(name='tfrecord_file', default='/media/ulsee/E/keypoint_subnet_tfrecord/coco_train2017-test.tfrecord',
                       help='output path you want to save tfrecord data file')

tf.flags.DEFINE_integer(name='img_num', default=21,
                        help='define how many images to build tfrecord data, zero menas all')


def img_reader(image_dir):
    '''
    read imgs in image_dir and return some lists
    :param image_dir: string, path of input image dir, e.g., /path/to/imgdir/
    :return:
    img_paths: img path for every single img
    img_ids: img name without suffix for every single img
    img_heights: img height for every single img
    img_widths: img width for every single img
    '''

    img_paths = []
    img_ids = []
    img_heights = []
    img_widths = []

    img_count = 0
    file_suffix = ['jpg', 'png']

    for img_file in os.scandir(image_dir):
        if FLAGS.img_num != 0 and img_count == FLAGS.img_num:
            break

        suffix = img_file.name[-3:].lower()

        if suffix in file_suffix and img_file.is_file() :

            img = cv2.imread(img_file.path, cv2.IMREAD_COLOR)
            height, width, channels = img.shape

            img_ids.append(img_file.name[:-4])
            img_paths.append(img_file.path)
            img_heights.append(height)
            img_widths.append(width)

            img_count += 1
            print ('------------------{}-----------------'.format(img_count))


    return img_paths, img_ids, img_heights, img_widths

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _strs_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def tfrecord_writer(img_dir, output_file):
    '''
    conver img in img_dir into tfrecord, saved as output_file
    :param img_dir: img directory
    :param output_file: tfrecord name with path to save
    :return:
    '''
    # print (1)
    img_paths, img_ids, img_heights, img_widths = img_reader(image_dir=img_dir)
    # print (2)
    output_dir = os.path.dirname(output_file)
    try:
        os.makedirs(output_dir)
    except os.error:
        pass

    img_nums = len(img_paths)

    writer = tf.python_io.TFRecordWriter(output_file)
    print('start writing tfrecord....')

    for i in range(img_nums):
        img_path = img_paths[i]
        img_id = bytes(img_ids[i], encoding='utf-8')
        img_height = img_heights[i]
        img_width = img_widths[i]

        with tf.gfile.FastGFile(img_path, 'rb') as f:
            img = f.read()

        example = tf.train.Example(features=tf.train.Features(
            feature={
                'image': _bytes_feature(img),
                'id': _strs_feature(img_id),
                'height': _int64_feature(img_height),
                'width': _int64_feature(img_width)
            }))
        writer.write(example.SerializeToString())

        if (i + 1) % 1000 == 0:
            print('processing....{}/{}'.format(i+1, img_nums))
    print ('tfrecord write done.')
    writer.close()

def main(argv):
    tfrecord_writer(FLAGS.image_dir, FLAGS.tfrecord_file)

if __name__ == '__main__':
    tf.app.run()