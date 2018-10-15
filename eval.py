# encoding: utf-8
'''
@author: shiwei hou
@contact: murdockhou@gmail.com
@software: PyCharm
@file: eval.py
@time: 18-9-29 下午6:56
'''

import tensorflow as tf

from keypoint_subnet.src.backbone import BackBone
from keypoint_subnet.src.model import Keypoint_Subnet

from person_detect.src.retinanet import RetinaNet
from person_detect.anchor.box_coder import FasterRCNNBoxCoder
from person_detect.anchor.anchor_generator import create_retinanet_anchors
from person_detect.person_detect_test import draw_boxs

from pose_residual_network.src.PRN import PRN

import numpy as np
import cv2, os, json, time
from datetime import datetime

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer('img_size', 480, '')
tf.flags.DEFINE_integer('batch_size', 1, '')
tf.flags.DEFINE_integer('num_keypoints', 17, '')
tf.flags.DEFINE_integer('num_classes', 1, '')
tf.flags.DEFINE_string('keypoint_trained_model', '', 'the checkpoint path of keypoint_subnet trained model.')
tf.flags.DEFINE_string('person_detect_trained_model', '', 'the checkpoint path of person_detect trained model.')
tf.flags.DEFINE_string('PRN trained model', '', 'the checkpoint path of PRN trained model.')

point_score_threshold = 0.05
point_nms_threshold   = 5

keypoint_checkpoint   = ''
person_checkpoint     = ''
prn_checkpoint        = ''



def prepare_heatmaps(pred_heatmap):
    for i in range(FLAGS.batch_size):
        current_pred_heatmap = pred_heatmap[i]
        for c in range(FLAGS.num_keypoints):
            current_channel = current_pred_heatmap[:, :, c]
            #-----------------find possible point location---------#
            threshold = point_score_threshold
            nms_thres = point_nms_threshold
            x, y      = np.where(current_channel > threshold)
            coordinate = list(zip(x, y))

            s = []
            for coor in coordinate:
                s.append(current_channel[coor])
            s = np.asarray(s)

            s_index = s.argsort()[::-1]  # 降序，第一个位置的索引值最大

            # nms
            keep_index = []

            while s_index.size > 0:
                keep_index.append(s_index[0])
                s_index = s_index[1:]
                last = []
                for index in s_index:
                    # print(keep[-1], index)
                    distance = np.sqrt(np.sum(np.square(
                        np.asarray(coordinate[keep_index[-1]]) - np.asarray(coordinate[index])
                    )))
                    if distance > nms_thres:
                        last.append(index)
                s_index = np.asarray(last)
            #--------------------change current_channel-------------#
            new_channel = np.zeros(current_channel.shape)
            for index in keep_index:
                coord = coordinate[index]
                new_channel[coord[0], coord[1]] = 1
            current_channel = new_channel

            current_pred_heatmap[:, :, c] = current_channel
        pred_heatmap[i] = current_pred_heatmap
    # ---------------resize pred_heatmap to img_size---------------#
    # pred_heatmap = np.sum(pred_heatmap, axis=3, keepdims=True)
    new_heatmap  = np.zeros((FLAGS.batch_size, FLAGS.img_size, FLAGS.img_size, FLAGS.num_keypoints), dtype=np.float32)
    for i in range(FLAGS.batch_size):
        current_pred_heatmap = pred_heatmap[i]
        for c in range(FLAGS.num_keypoints):
            current_channel = current_pred_heatmap[:,:,c]
            current_channel = np.expand_dims(current_channel, axis=2)
            current_channel = cv2.resize(current_channel, (FLAGS.img_size, FLAGS.img_size), interpolation=cv2.INTER_NEAREST)
            current_channel = np.squeeze(current_channel, axis=2)
            current_pred_heatmap[:, :, c] = current_channel
        new_heatmap[i]       = current_pred_heatmap
    return new_heatmap

def select_pred_boxs(loc_preds, cls_preds):

    # -------------------------(1) generate anchor-----------------------------------------#
    input_size = [tf.to_float(FLAGS.img_size), tf.to_float(FLAGS.img_size)]
    feature_map_list = [(tf.ceil(tf.multiply(input_size[0], 1 / pow(2., i + 3))),
                         tf.ceil(tf.multiply(input_size[1], 1 / pow(2., i + 3))))
                        for i in range(5)]
    anchor_generator = create_retinanet_anchors()
    anchor = anchor_generator.generate(input_size, feature_map_list)

    # --------------------------(2)-decode loc_pred---------------------------------------#
    current_loc_pred = loc_preds[0]
    # 根据anchor将网络的loc输出解码，表示为[ymin, xmin, ymax, xmax]
    current_box_list = FasterRCNNBoxCoder().decode(current_loc_pred, anchor.get())
    current_decoded_loc_pred = current_box_list.get()
    # ---------------------------(3)------NMS---------------------------------------------#
    box_score = tf.nn.softmax(cls_preds[0])
    box_score = box_score[:, 1]
    top_k_score, top_k_indices = tf.nn.top_k(box_score, k=60)
    top_k_boxs = tf.gather(current_decoded_loc_pred, top_k_indices)
    nms_indices = tf.image.non_max_suppression(boxes=top_k_boxs, scores=top_k_score, max_output_size=5,
                                               iou_threshold=0.5)
    final_boxs = tf.gather(top_k_boxs, nms_indices)
    final_scores = tf.gather(top_k_score, nms_indices)

    return final_boxs, final_scores

def make_prn_inputs(heatmap, boxs):
    '''

    :param heatmap: [img_size, img_size, 17]
    :param boxs: [-1, 4],(xmin, ymin, xmax, ymax)
    :return:
        prn_inputs:[num_boxs, 56, 36, 17]
    '''
    box_relative_position = boxs
    prn_inputs = []
    for box in list(boxs):
        xmin = int(box[0])
        ymin = int(box[1])
        xmax = int(box[2])
        ymax = int(box[3])
        box_area = heatmap[xmin:xmax, ymin:ymax, :]
        for c in range(FLAGS.num_keypoints):
            current_channel = box_area[:, :, c]
            current_channel = np.expand_dims(current_channel, axis=2)
            current_channel = cv2.resize(current_channel, (56,36),
                                         interpolation=cv2.INTER_NEAREST)
            current_channel = np.squeeze(current_channel, axis=2)
            box_area[:, :, c] = current_channel
        # box_area = cv2.resize(box_area, (56,36), interpolation=cv2.INTER_NEAREST)
        prn_inputs.append(box_area)

    prn_inputs = np.asarray(prn_inputs)
    return prn_inputs, box_relative_position

def get_keypoints(prn_out, ori_box):
    keypoints = []
    # 找到PRN网络输出的结果中，合适的关键点的坐标在哪
    for c in range(FLAGS.num_keypoints):
        current_channel = prn_out[:, :, c]
        # -----------------find possible point location---------#
        threshold = point_score_threshold
        nms_thres = point_nms_threshold
        x, y = np.where(current_channel > threshold)
        coordinate = list(zip(x, y))

        s = []
        for coor in coordinate:
            s.append(current_channel[coor])
        s = np.asarray(s)

        s_index = s.argsort()[::-1]  # 降序，第一个位置的索引值最大

        # nms
        keep_index = []

        while s_index.size > 0:
            keep_index.append(s_index[0])
            s_index = s_index[1:]
            last = []
            for index in s_index:
                # print(keep[-1], index)
                distance = np.sqrt(np.sum(np.square(
                    np.asarray(coordinate[keep_index[-1]]) - np.asarray(coordinate[index])
                )))
                if distance > nms_thres:
                    last.append(index)
            s_index = np.asarray(last)

        for index in keep_index:
            coord = coordinate[index]
            prn_out_x = coord[0]
            prn_out_y = coord[1]
            ori_x     = prn_out_x + ori_box[0]
            ori_y     = prn_out_y + ori_box[1]
            keypoint  = [ori_x, ori_y]
            keypoints.append(keypoint)

    return keypoints


def eval():
    tf.reset_default_graph()
    graph = tf.Graph()
    with graph.as_default():
        #----------------------------------net work-----------------------------------------#
        #--------------------------------1. backbone----------------------------------------#
        backbone              = BackBone(img_size=FLAGS.img_size, batch_size=FLAGS.batch_size, is_training=False)
        fpn, feature_map_dict = backbone.build_fpn_feature()
        #--------------------------------2. keypoint net-------------------------------------#
        keypoint_net    = Keypoint_Subnet(inputs=backbone.input_imgs, img_size=FLAGS.img_size, fpn=fpn, num_classes=FLAGS.num_keypoints,
                                          batch_size=FLAGS.batch_size)
        pred_heatmap, _ = keypoint_net.forward()
        #--------------------------------3. person detect------------------------------------#
        retina_net           = RetinaNet(fpn=fpn, feature_map_dict=feature_map_dict, batch_size=FLAGS.batch_size,
                                       num_classes=FLAGS.num_classes+1, is_training=False)
        loc_preds, cls_preds = retina_net.forward()

        #-----------------------------select pred_box-----------------------------------------#
        select_boxs, select_scores = select_pred_boxs(loc_preds, cls_preds) # select_boxs, (-1, 4), select_scores(-1,)

        # --------------------------------4. pose-residual-net--------------------------------#
        inputs  = tf.placeholder(tf.float32, shape=(1, 56, 36, 17), name='inputs')
        prn     = PRN(inputs=inputs, output_node=1*56*36*17, is_training=False)
        prn_out = prn.forward()

        #---------------------------------restore------------------------------------#
        res50_var_list           = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='resnet_v2_50')
        fpn_var_list             = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='build_fpn_feature')
        keypoint_subnet_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='keypoint_subnet')
        person_detect_var_list   = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='retina_net')
        prn_var_list             = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='pose-residual-network')

        global_list     = tf.global_variables()
        bn_moving_vars  = [g for g in global_list if 'moving_mean' in g.name]
        bn_moving_vars += [g for g in global_list if 'moving_variance' in g.name]

        keypoint_restore = tf.train.Saver(var_list=(res50_var_list+bn_moving_vars+fpn_var_list+keypoint_subnet_var_list))
        person_restore   = tf.train.Saver(var_list=person_detect_var_list)
        prn_restore      = tf.train.Saver(var_list=prn_var_list)


        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        config  = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(graph=graph, config=config) as sess:
            sess.run(init_op)
            keypoint_restore.restore(sess, keypoint_checkpoint)
            person_restore.restore(sess, person_checkpoint)
            prn_restore.restore(sess, prn_checkpoint)

            # TODO : read img
            img_ori = cv2.imread('', cv2.IMREAD_COLOR)
            img_copy = img_ori.copy()
            img = cv2.resize(img_copy, (FLAGS.img_size, FLAGS.img_size), interpolation=cv2.INTER_NEAREST)

            heatmaps, boxs, scores = sess.run([pred_heatmap, select_boxs, select_scores],
                                              feed_dict={backbone.input_imgs: [img]})
            # ------------------------------prepared pred_heatmap to img_size------------------------#
            _heatmap = prepare_heatmaps(heatmaps)  # pred_heatmap, (batch_size, img_size, img_size, 17)
            # ---------------------------combine heatmap and boxs----------------------------------#
            prn_inputs, box_relative_position = make_prn_inputs(_heatmap[0], boxs)

            for i in range(prn_inputs.shape[0]):
                single_input = prn_inputs[i]
                single_out   = sess.run(prn_out, feed_dict={inputs:[single_input]})
                # TODO: 修改single_out，使其返回到其在原来图片中的位置
                keypoints    = get_keypoints(single_out, box_relative_position[i])
                # draw box and keypoint on img
                box   = box_relative_position[i]
                score = scores[i]
                cv2.rectangle(img, (box[1], box[0]), (box[3], box[2]), (255 - 10 * i, 0, 0), 1)
                cv2.putText(img, 'person: ' + str(score[i]), (box[1], box[0]), cv2.FONT_HERSHEY_COMPLEX, 1,
                            (0, 0, 255), 1)
                for point in keypoints:
                    cv2.circle(img, (point[1], point[0]), 5, (0, 0, 255), -1)

            img = cv2.resize(img, (img_ori.shape[0], img_ori.shape[1]), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite('result.jpg', img)

if __name__ == '__main__':
    eval()


