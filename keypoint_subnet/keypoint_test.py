# encoding: utf-8
'''
@author: shiwei hou
@contact: murdockhou@gmail.com
@software: PyCharm
@file: keypoint_test.py
@time: 18-10-8 上午9:50
'''
import tensorflow as tf
from datetime import datetime
import os, cv2, json
import logging
import numpy as np
import math

import sys

from src.backbone import BackBone
from src.model import Keypoint_Subnet
from src.get_heatmap import get_heatmap
from src.reader import Keypoint_Reader
from src.json_read import  load_json, load_coco_json
from src.img_pre_processing import image_vertical_flipping

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('model', '/media/ulsee/D/keypoint_subnet/20181023-2043/model_alter.ckpt-239999',
                       'model path you want to test, e.g., (/media/ulsee/D/multi-pose-net/20180829-1927/model.ckpt-xxxxx)')
tf.flags.DEFINE_string('img_path', '/media/ulsee/E/datasets/coco/cocotrain2017',
                       'image path to test model.')
tf.flags.DEFINE_string('save_path', '/media/ulsee/E/keypoint/coco/train2017', 'path to save image test result')
tf.flags.DEFINE_boolean('is_training', True, '')
tf.flags.DEFINE_integer(name='batch_size', default=1, help='train batch size number')
tf.flags.DEFINE_integer(name='img_size', default=480, help='net input size')
tf.flags.DEFINE_integer(name='num_keypoints', default=17, help='number of keypoints to detect')



def is_image(img_name):
    img_name = img_name.lower()
    if img_name.endswith('.jpg') or img_name.endswith('.png') or img_name.endswith('jpeg'):
        return True
    return False



def deal_with_heatmaps(img, heatmap, factorx, factory, num_keypoints, score_threshold, nms_threshold=5, type=1):
    '''

    :param img:
    :param heatmap:
    :param num_keypoints:
    :param type: 1 for single person and other for multi-person
    :return:
    '''
    if type == 1:
        for c in range(num_keypoints):
            current_heatmap = heatmap[0, :, :, c]

            cur_max = np.max(current_heatmap)
            # print (cur_max)
            if cur_max < score_threshold:
                continue
            index_all = np.where(current_heatmap == cur_max)
            coorx = index_all[0][0]
            coory = index_all[1][0]

            coorx = int(coorx * factorx)
            coory = int(coory * factory)

            cv2.circle(img, (coory, coorx), 5, (0, 0, 255), -1)
            cv2.putText(img, str(c), (coory, coorx), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 1)
    else:
        threshold = score_threshold
        nms_threshold = nms_threshold
        cur_max = 0
        count = 0
        for c in range(num_keypoints):
            current_heatmap = heatmap[0, :, :, c]
            x, y = np.where(current_heatmap > threshold)
            coordinate = list(zip(x, y))
            # print(coordinate)
            s = []
            for coor in coordinate:
                # print(coor)
                # print(current_heatmap[coor])
                s.append(current_heatmap[coor])
            s = np.asarray(s)
            # print(s)
            s_index = s.argsort()[::-1] # 降序，第一个位置的索引值最大
            # print(s_index)
            # nms
            keep = []

            while s_index.size > 0:
                keep.append(s_index[0])
                s_index = s_index[1:]
                last = []
                for index in s_index:
                    # print(keep[-1], index)
                    distance = np.sqrt(np.sum(np.square(
                        np.asarray(coordinate[keep[-1]]) - np.asarray(coordinate[index])
                    )))
                    if distance > nms_threshold:
                        last.append(index)

                s_index = np.asarray(last)

            for index in keep:
                coor = coordinate[index]
                coorx = coor[0]
                coory = coor[1]

                coorx = int(coorx * factorx)
                coory = int(coory * factory)

                cv2.circle(img, (coory, coorx), 5, (0, 0, 255), -1)
                cv2.putText(img, str(c), (coory, coorx), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 1)
                count += 1
                cur_max += s[index]

        cur_max = cur_max / (count if count > 0 else 1)

    return img, cur_max

def _test(score_threshold, nms_threshold):
    global  save_json
    if not os.path.exists(FLAGS.save_path):
        os.makedirs(FLAGS.save_path)

    graph = tf.Graph()
    with graph.as_default():
        # ------------------------get backbone net--------------------------------#
        backbone = BackBone(img_size=FLAGS.img_size, batch_size=FLAGS.batch_size, is_training=FLAGS.is_training)
        fpn, _   = backbone.build_fpn_feature()
        # ---------------------------keypoint net---------------------------------#
        keypoint_net = Keypoint_Subnet(inputs=backbone.input_imgs, img_size=backbone.img_size, fpn=fpn,
                                       batch_size=backbone.batch_size, num_classes=FLAGS.num_keypoints)
        pre_heat, _ = keypoint_net.forward()

        g_list = tf.global_variables()

        bn_moving_mean = [g for g in g_list if 'moving_mean' in g.name]
        bn_moving_vars = [g for g in g_list if 'moving_variance' in g.name]

        var_list = tf.trainable_variables()
        var_list += bn_moving_vars + bn_moving_mean
        # for var in var_list:
        #     print (var)

        init_op = tf.group(tf.global_variables_initializer())

        saver   = tf.train.Saver()

        with tf.Session(graph=graph) as sess:
            sess.run(init_op)
            saver.restore(sess, FLAGS.model)
            print('model restore successfully.')

            img_num = 0
            test_img_id = ['000000135361','000000265513','000000496607','000000270836']

            avg = 0

            for img in os.listdir(FLAGS.img_path):
                # if not is_image(img):
                #     continue
                # if img.split('.')[0] not in test_img_id:
                #     continue
                img_num += 1
                img_ori = cv2.imread(os.path.join(FLAGS.img_path, img), cv2.IMREAD_COLOR)

                # img_ori = cv2.flip(img_ori, 1)

                img_copy = img_ori.copy()

                # img_input = img_copy
                img_input = cv2.resize(img_copy, (FLAGS.img_size, FLAGS.img_size), interpolation=cv2.INTER_NEAREST)
                heatmaps = sess.run(pre_heat,
                                              feed_dict={backbone.input_imgs:[img_input]})

                factorx = img_ori.shape[0] / heatmaps.shape[1]
                facotry = img_ori.shape[1] / heatmaps.shape[2]
                img_save, cur_max = deal_with_heatmaps(img_ori, heatmaps, factorx, facotry, FLAGS.num_keypoints,
                                                       score_threshold=score_threshold, nms_threshold=nms_threshold, type=2)
                avg += cur_max
                cv2.imwrite(os.path.join(FLAGS.save_path, img), img_save)
                # for mean in bn_moving_vars:
                #     print(sess.run(mean))
                #     break

                if img_num == 400:
                    break
                print('tested {}'.format(img_num))

            print('avg max === {}'.format(avg/img_num))

if __name__ == '__main__':
    _test(score_threshold=0.05, nms_threshold=5)