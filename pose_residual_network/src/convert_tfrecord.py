# encoding: utf-8
'''
@author: shiwei hou
@contact: murdockhou@gmail.com
@software: PyCharm
@file: convert_tfrecord.py
@time: 18-9-27 下午3:15
'''
import tensorflow as tf
import cv2, os, json
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import gaussian


FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('json_file', '/media/ulsee/E/pose_residual_net_tfrecord/cocotrain2017_convert_ai.json',
                       '')
tf.flags.DEFINE_string('tfrecord_file', '/media/ulsee/E/pose_residual_net_tfrecord/coco_train2017.tfrecord',
                       'tfrecord file')
tf.flags.DEFINE_integer('height', 56, 'prn net input height')
tf.flags.DEFINE_integer('width', 36, 'prn net input width')
tf.flags.DEFINE_integer('channels', 17, 'number of keypoints')

def _int64_feature(value):
    ''' Wrapper for inserting int64 feature into Example proto'''
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list = tf.train.Int64List(value=value))

def _float_feature(value):
    ''' Wrapper for inserting float feature into Example proto'''
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _bytes_feature(value):
    ''' Wrapper for inserting bytes feature into Example proto'''
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list = tf.train.BytesList(value=value))

def _string_feature(value):
    ''' Wrapper for inserting string (actually bytes) feature into Example proto'''
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list = tf.train.BytesList(value=value))

def convert_to_tfrecord(json_file, tfrecord_file):

    f      = open(json_file, encoding='utf-8')
    labels = json.load(f)

    if isinstance(labels, dict):
        pass
    elif isinstance(labels, list):
        convert_ai_challenger(labels, tfrecord_file)
    else:
        raise ValueError('Json file format is wrong!!!')


def convert_ai_challenger(labels, tfrecord_file):

    tfrecord_dir = os.path.dirname(tfrecord_file)
    if not os.path.exists(tfrecord_dir):
        os.makedirs(tfrecord_dir)

    writer     = tf.python_io.TFRecordWriter(tfrecord_file)
    total_imgs = len(labels)
    deal_imgs  = 0
    useless    = 0
    for label in labels:
        # print (label['image_id'])

        kp_anno       = label['keypoint_annotations']
        human_anno    = label['human_annotations']
        humans        = kp_anno.keys()
        all_keypoints = [kp for kp in kp_anno.values()]

        for human in humans:
            kp  = kp_anno[human]
            kpv = kp[2::3]
            if np.sum(kpv>0) < 4:
                useless += 1
                continue
            box = human_anno[human]
            box[2] = box[2] - box[0]
            box[3] = box[3] - box[1]

            if box[2] == 0 or box[3] == 0:
                continue

            tf_label  = get_label_for_single_box(kp, box)
            tf_inputs = get_input_for_single_box(all_keypoints, box)

            #
            # img1 = np.sum(tf_label, axis=2, keepdims=True)
            # cv2.imwrite('label.jpg', img1*255)
            # img2 = np.sum(tf_inputs, axis=2, keepdims=True)
            # cv2.imwrite('input.jpg', img2*255)
            #

            example = tf.train.Example(features=tf.train.Features(
                feature = {
                    'input':_float_feature(list(np.reshape(np.asarray(tf_inputs, dtype=np.float32), (-1, )))),
                    'label':_float_feature(list(np.reshape(np.asarray(tf_label, dtype=np.float32), (-1, ))))
                }
            ))

            writer.write(example.SerializeToString())
        deal_imgs += 1

        # if deal_imgs == 2:
        #     break

        if deal_imgs % 1000 == 0:
            print ('Processing {}/{}'.format(deal_imgs, total_imgs))
            print ('Useless boxs {}'.format(useless))

    writer.close()
    print ('Converting tf record done.')


def get_label_for_single_box(keypoints, bbox):
    label = np.zeros((FLAGS.height, FLAGS.width, FLAGS.channels))

    x = int(bbox[0])
    y = int(bbox[1])
    w = float(bbox[2])
    h = float(bbox[3])

    x_scale = float(FLAGS.width) / w
    y_scale = float(FLAGS.height) / h

    kpx = keypoints[0::3]
    kpy = keypoints[1::3]
    kpv = keypoints[2::3]

    for j in range(FLAGS.channels):
        if kpv[j] != 3 and kpv[j] != 0:
            x0 = int((kpx[j] - x) * x_scale)
            y0 = int((kpy[j] - y) * y_scale)

            if x0 >= FLAGS.width and y0 >= FLAGS.height:
                label[FLAGS.height-1, FLAGS.width-1, j] = 1
            elif x0 >= FLAGS.width:
                try:
                    label[y0, FLAGS.width-1, j] = 1
                except:
                    label[0,  FLAGS.width-1, j] = 1
            elif y0 >= FLAGS.height:
                try:
                    label[FLAGS.height-1, x0, j] = 1
                except:
                    label[FLAGS.height-1, 0,  j] = 1
            elif x0 < 0 and y0 < 0:
                label[0, 0, j]   = 1
            elif x0 < 0:
                label[y0, 0, j]  = 1
            elif y0 < 0:
                label[0, x0, j]  = 1
            else:
                label[y0, x0, j] = 1

    # for c in range(FLAGS.channels):
    #     label[:, :, c] = gaussian(label[:, :, c],sigma=0.5)
    label = gaussian(label, sigma=2, mode='constant', multichannel=True)
    return label

def get_input_for_single_box(keypoints, bbox):
    inputs    = np.zeros((FLAGS.height, FLAGS.width, FLAGS.channels))
    threshold = 0.21

    x = int(bbox[0])
    y = int(bbox[1])
    w = float(bbox[2])
    h = float(bbox[3])


    x_scale = float(FLAGS.width) / w
    y_scale = float(FLAGS.height) / h

    for ann in keypoints:
        kpx = ann[0::3]
        kpy = ann[1::3]
        kpv = ann[2::3]


        for j in range(FLAGS.channels):
            if kpv[j] != 3 and kpv[j] != 0:
                if kpx[j] > bbox[0] - bbox[2] * threshold and kpx[j] < bbox[0] + bbox[2] * (1 + threshold):
                    if kpy[j] > bbox[1] - bbox[3] * threshold and kpy[j] < bbox[1] + bbox[3] * (1 + threshold):

                        x0 = int((kpx[j] - x) * x_scale)
                        y0 = int((kpy[j] - y) * y_scale)

                        if x0 >= FLAGS.width and y0 >= FLAGS.height:
                            inputs[FLAGS.height - 1, FLAGS.width - 1, j] = 1
                        elif x0 >= FLAGS.width:
                            try:
                                inputs[y0, FLAGS.width - 1, j] = 1
                            except:
                                inputs[0, FLAGS.width - 1, j] = 1
                        elif y0 >= FLAGS.height:
                            try:
                                inputs[FLAGS.height - 1, x0, j] = 1
                            except:
                                inputs[FLAGS.height - 1, 0, j] = 1
                        elif x0 < 0 and y0 < 0:
                            inputs[0, 0, j] = 1
                        elif x0 < 0:
                            inputs[y0, 0, j] = 1
                        elif y0 < 0:
                            inputs[0, x0, j] = 1
                        else:
                            inputs[y0, x0, j] = 1


    for c in range(FLAGS.channels):
        inputs[:, :, c] = gaussian(inputs[:, :, c])
    return inputs


if __name__ == '__main__':
    convert_to_tfrecord(FLAGS.json_file, FLAGS.tfrecord_file)



