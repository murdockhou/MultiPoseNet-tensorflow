# encoding: utf-8
'''
@author: shiwei hou
@contact: murdockhou@gmail.com
@software: PyCharm
@file: multi_pose_net_eval.py
@time: 18-10-19 上午9:30
'''
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from keypoint_subnet.src.backbone import BackBone
from keypoint_subnet.src.model import Keypoint_Subnet
from keypoint_subnet.src.get_heatmap import get_single_heatmap

import sys
sys.path.append('./person_detect/')

from person_detect.src.retinanet import RetinaNet
from person_detect.anchor.anchor_generator import create_retinanet_anchors
from person_detect.anchor.box_coder import FasterRCNNBoxCoder
from pose_residual_network.src.PRN import PRN

import tensorflow as tf
import numpy as np
import os
import json
import cv2
import math
from skimage.filters import gaussian

tf.summary.image
# yolo v3
from yolo_v3.models.yolo_v3 import yolo_v3
from yolo_v3.utils.utils import process_image, get_anchors, get_classes, convert_box_coordinates, non_max_suppression, draw_boxes


json_file       = '/media/ulsee/E/coco_val2017_aiformat2.json'
coco_json_file  = '/media/ulsee/E/datasets/coco/annotations2017/person_keypoints_val2017.json'
img_path        = '/media/ulsee/E/datasets/coco/cocoval2017'
num_keypoints   = 17
num_classes     = 1
img_size        = 480
point_score     = 0.05
point_nms_thres = 5

keypoint_checkpoint     = '/media/ulsee/D/keypoint_subnet/20181015-1711/model_alter.ckpt-69999/model_alter.ckpt-339999'
person_detec_checkpoint = '/media/ulsee/D/retinanet/20181022-1738/model_alter.ckpt-16'
prn_checkpoint          = '/media/ulsee/D/PRN/20181015-0750/model.ckpt-245572'

# yolo v3
yolo_v3_checkpoint      = '/media/ulsee/D/yolov3/coco_pretrained_weights.ckpt'
yolo_height = 416
yolo_width  = 416
yolo_anchors = get_anchors('yolo_v3/utils/anchors.txt')
yolo_classes = get_classes('yolo_v3/utils/coco_classes.txt')


is_used_gt_box    = True
is_used_gt_points = True
is_used_yolo_boxs = False

def multi_pose_net_eval():
    '''
    对multipose net在coco val 2017上进行测试。注意程序中出现的所有坐标值都是相对于图片而言的，即[x,y]为图片上的[width, height]，
    在python中，转为矩阵操作后，实为ndarray上的[y, x]，即先行后列，图片上则是先列后行。
    :return:
    '''
    tf.reset_default_graph()
    graph = tf.Graph()

    with graph.as_default():
        backbone = BackBone(img_size=img_size, batch_size=1, is_training=False)
        fpn, fmd = backbone.build_fpn_feature()

        keypoint_net = Keypoint_Subnet(inputs=backbone.input_imgs, img_size=img_size, fpn=fpn, num_classes=num_keypoints, batch_size=1)
        pred_heatmap, _ = keypoint_net.forward()

        retina_net = RetinaNet(fpn=fpn, feature_map_dict=fmd, batch_size=1, num_classes=num_classes+1, is_training=False)
        loc_preds, cls_preds = retina_net.forward()
        select_boxs, select_scores = select_pred_boxs(loc_preds, cls_preds)  # select_boxs, (-1, 4), select_scores(-1,)

        prn_inputs_placeholder = tf.placeholder(tf.float32, shape=(1, 56, 36, 17), name='prn_inputs')
        prn = PRN(inputs=prn_inputs_placeholder, output_node=1 * 56 * 36 * 17, is_training=False)
        prn_out = prn.forward()

        yolo_v3_x = tf.placeholder(tf.float32, shape=[1, yolo_height, yolo_width, 3])
        yolo_v3_outputs = yolo_v3(inputs=yolo_v3_x, num_classes=len(yolo_classes), anchors=yolo_anchors, h=yolo_height, w=yolo_width, training=False)
        yolo_v3_raw_outptus = tf.concat(yolo_v3_outputs, axis=1)

        # ---------------------------------restore------------------------------------#
        res50_var_list           = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='resnet_v2_50')
        fpn_var_list             = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='build_fpn_feature')
        keypoint_subnet_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='keypoint_subnet')
        person_detect_var_list   = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='retina_net')
        prn_var_list             = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='pose-residual-network')

        global_list     = tf.global_variables()
        bn_moving_vars  = [g for g in global_list if 'moving_mean' in g.name]
        bn_moving_vars += [g for g in global_list if 'moving_variance' in g.name]

        res_moving_vars  = [g for g in bn_moving_vars if 'resnet_v2_50' in g.name]
        yolo_moving_vars = list(set(bn_moving_vars).difference(set(res_moving_vars)))
        # print (len(yolo_moving_vars))

        train_vars = tf.trainable_variables()
        yolo_train_vars = list(set(train_vars).difference(set(res50_var_list+fpn_var_list+keypoint_subnet_var_list+person_detect_var_list+prn_var_list)))
        # for var in yolo_train_vars:
        #     print (var)
        # print (len(yolo_train_vars))

        keypoint_restore = tf.train.Saver(
            var_list=(res50_var_list + res_moving_vars + fpn_var_list + keypoint_subnet_var_list))
        person_restore = tf.train.Saver(var_list=person_detect_var_list)
        prn_restore = tf.train.Saver(var_list=prn_var_list)
        yolo_restore = tf.train.Saver(var_list=(yolo_train_vars+yolo_moving_vars))

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(graph=graph, config=config) as sess:
            sess.run(init_op)
            keypoint_restore.restore(sess, keypoint_checkpoint)
            print ('#---------------keypoint subnet restored successfully.-------------#')
            person_restore.restore(sess, person_detec_checkpoint)
            print ('#---------------person detection subnet restored successfully.---------------#')
            prn_restore.restore(sess, prn_checkpoint)
            print ('#---------------prn subnet restored successfully.----------------#')
            yolo_restore.restore(sess, yolo_v3_checkpoint)
            print ('#---------------yolo v3 restored successfully----------------#')

            img_names, img_keypoints, img_boxs, img_ids = read_json(json_file)

            if len(img_names) != len(img_keypoints) and len(img_names) != len(img_boxs):
                print('ids, points, boxs not equal.')
                return

            my_results = []
            coco_test_img_ids = []
            for i in range(len(img_names)):
                img_name = img_names[i]
                img_id   = img_ids[i]

                img_file = os.path.join(img_path, img_name+'.jpg')

                img = cv2.imread(img_file, cv2.IMREAD_COLOR)
                img_copy = img.copy()

                height, width, channels = img_copy.shape
                # print ('ori shpae = ', img_copy.shape)

                img_input = cv2.resize(img_copy, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
                yolo_img_input = cv2.resize(img_copy, (yolo_width, yolo_height)) / 255.0

                yolo_output = sess.run(yolo_v3_raw_outptus, feed_dict={yolo_v3_x:[np.array(yolo_img_input, dtype=np.float32)]})

                heatmaps, boxs, scores = sess.run([pred_heatmap, select_boxs, select_scores],
                                                  feed_dict={backbone.input_imgs:[img_input]})

                net_keypoints = prepare_heatmaps(heatmaps, height, width)
                net_boxs      = translate(boxs, height, width) #[-1, 4], [xmin, xmax, ymin, ymax]

                yolo_boxs     = get_yolo_boxs(yolo_output, img.shape, yolo_img_input.shape)

                gt_boxs = np.asarray(img_boxs[i])
                # TODO gt部分都整完了，剩下预测的部分，注意全部弄成ndarray形式，list容易出错，由于浅拷贝的原因
                prn_boxs   = net_boxs
                prn_points = net_keypoints

                if is_used_gt_box:
                    prn_boxs = gt_boxs

                if is_used_gt_points:
                    prn_points = img_keypoints[i]

                if is_used_yolo_boxs:
                    prn_boxs = yolo_boxs

                #--------------------------------------------------------#
                # img_copy2 = img.copy()
                # for p in range(net_keypoints.shape[0]):
                #     kps = net_keypoints[p]
                #     for cc in range(kps.shape[0]):
                #         x = int(kps[cc][0])
                #         y = int(kps[cc][1])
                #         cv2.circle(img_copy2, (x, y), 5, (0, 0, 255), -1)
                #         cv2.putText(img_copy2, str(cc), (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 1)
                # cv2.imwrite('net_points.jpg', img_copy2)
                #--------------------------------------------------------#

                prn_inputs = make_prn_inputs(prn_points, prn_boxs)
                prn_inputs = np.asarray(prn_inputs)

                for batch in range(prn_inputs.shape[0]):
                    for c in range(num_keypoints):
                        prn_inputs[batch, :, :, c] = gaussian(prn_inputs[batch, :, :, c])

                prn_keypoints_outputs = []
                json_scores = []
                for batch in range(prn_inputs.shape[0]):
                    single_prn_out = sess.run(prn_out, feed_dict={prn_inputs_placeholder:[prn_inputs[batch]]})
                    single_prn_out = np.reshape(single_prn_out, newshape=(56, 36, 17))
                    single_box_keypoints, single_pose_score = get_box_keypoints(single_prn_out, 56, 36)
                    prn_keypoints_outputs.append(single_box_keypoints)
                    json_scores.append(single_pose_score)
                # print ('prn output keypoints = {}'.format(prn_keypoints_outputs))
                json_boxs, json_keypoints = get_ori_location(prn_keypoints_outputs, prn_boxs)
                json_scores = np.asarray(json_scores)

                #---------------------------------------------------------#
                # n = json_boxs.shape[0]
                # json_boxs = np.zeros((n, 4))
                # json_keypoints = np.zeros((n, 51))
                # for b in range(n):
                #     json_boxs[b] = gt_boxs[b]
                #     json_keypoints[b] = img_keypoints[i][b]
                # json_keypoints = np.reshape(json_keypoints, (n, 17, 3))
                # print ('json_keypoints = {}'.format(json_keypoints))
                # --------------------------------------------------------#
                # img_copy3 = img.copy()
                # for p in range(json_keypoints.shape[0]):
                #     kps = json_keypoints[p]
                #     for cc in range(kps.shape[0]):
                #         x = int(kps[cc][0])
                #         y = int(kps[cc][1])
                #         cv2.circle(img_copy3, (x, y), 5, (0, 0, 255), -1)
                #         cv2.putText(img_copy3, str(cc), (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 1)
                # cv2.imwrite('prn_points.jpg', img_copy3)
                # --------------------------------------------------------#

                # print (json_keypoints)
                if json_keypoints.shape[0] != json_boxs.shape[0]:
                    print ('prn_box and prn_keypoints should be equal firt shape.')
                    return

                img_id = int(img_id)
                is_test_img_id = True

                for batch in range(json_boxs.shape[0]):
                    if json_boxs[batch][2] == 0 or json_boxs[batch][3] == 0:
                        continue
                    k = np.zeros(51)
                    k[0::3] = json_keypoints[batch, :, 0]
                    k[1::3] = json_keypoints[batch, :, 1]
                    k[2::3] = [2] * 17
                    image_data = {
                        'image_id': img_id,
                        'bbox': list(json_boxs[batch]),
                        'score': json_scores[batch],
                        'category_id':1,
                        'keypoints': k.tolist()
                    }
                    is_test_img_id = True

                    my_results.append(image_data)

                if is_test_img_id:
                    coco_test_img_ids.append(img_id)

                print ('Processing {}/{}'.format(i, len(img_names)))
                # if i == 0:
                #     break

                # break

            ann_filename = 'val2017_PRN_keypoint_results_mine.json'
            # write output
            json.dump(my_results, open(ann_filename, 'w'), indent=4)
            coco = COCO('/media/ulsee/E/datasets/coco/annotations2017/person_keypoints_val2017.json')
            # load results in COCO evaluation tool
            coco_pred = coco.loadRes(ann_filename)

            # run COCO evaluation
            coco_eval = COCOeval(coco, coco_pred, 'keypoints')
            coco_eval.params.imgIds = coco_test_img_ids
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()

def get_ori_location(prn_keypoints, prn_boxs):
    '''

    :param prn_keypoints: 列表的列表的列表，每个列表都是一个包含17的列表的列表,转成为ndarray后，shape为[-1, 17, 2]，
                          是根据PRN网络的得到的每个box内部包含的关键点的位置，是相对于输入大小为（56,36,17）的box的位置， 格式为[[x1, y1], [x2, y2]...[x17, y17]]
    :param prn_boxs: 列表的列表，每个小列表都是一个box, box 格式为[xmin, ymin, xmax, ymax], 坐标是和原始图像大小相比，没有经过任何缩放操作
    :return:
        json_boxs: ndarray, [-1, 4]包含所有的框在原来图片上的位置，转化为coco数据格式[xmin, ymin, w, h]
        json_keypoints: ndarray, [-1, 17, 2]  包含这个图片上所有的关键点在图片上原来的位置, [ [17*2], [17*2], ]
    '''
    box_x_scale = 1.
    box_y_scale = 1.

    json_boxs = []
    json_keypoints = []

    prn_boxs = list(prn_boxs)
    for i in range(len(prn_boxs)):
        box = prn_boxs[i]
        # print (box)

        kps = prn_keypoints[i]

        new_kps = []
        # 先找到keypoints相对于box这么大的box应该所在的位置是哪个
        # 再根据box的值，找到此时keypoints相对于img_size这么大所在的位置
        # 最后再找到keypoints相对于原始图片所在的位置
        for c in range(num_keypoints):

            point_x = kps[c][0]
            point_y = kps[c][1]
            # print ('[x, y] === ', [point_x, point_y])
            box_height = box[3] - box[1]
            box_width  = box[2] - box[0]

            point_x_box = point_x * (box_width/36)
            point_y_box = point_y * (box_height/56)

            point_x_img_size = point_x_box + box[0]
            point_y_img_size = point_y_box + box[1]

            point_x_ori = point_x_img_size * box_x_scale
            point_y_ori = point_y_img_size * box_y_scale
            # print ([point_x_ori, point_y_ori])

            new_kps.append([point_x_ori, point_y_ori])

        json_keypoints.append(new_kps)

        ori_box = [1, 2, 3, 4]
        ori_box[0] = box[0] * box_x_scale
        ori_box[1] = box[1] * box_y_scale
        ori_box[2] = box[2] * box_x_scale
        ori_box[3] = box[3] * box_y_scale
        ori_box[2] = ori_box[2] - ori_box[0]
        ori_box[3] = ori_box[3] - ori_box[1]
        json_boxs.append(ori_box)

    return  np.asarray(json_boxs), np.asarray(json_keypoints)


def get_box_keypoints(prn_out, ori_height, ori_width):
    '''
    get single output for each channel.
    :param prn_out: a heatmap, typically the prn net work output
    :return:
        keypoints: a list of list, contains one pair coordinate for each channel, e.g.,[ [x1, y1], [x2, y2],...,[x17, y17]]
    '''
    keypoints = []
    pose_score = 0
    for c in range(num_keypoints):
        xscore = yscore = 0.
        current_channel = prn_out[:, :, c]
        cur_max = np.max(current_channel)
        if cur_max == 0:
            coorx = 0.
            coory = 0.
            score = 0.
        else:
            index_all = np.where(current_channel == cur_max)
            coory = index_all[0][0]
            coorx = index_all[1][0]

            # print ('1---(x, y) == ', (coorx, coory))

            # thres = 10
            # xmin  = max(0, coorx-thres)
            # ymin  = max(0, coory-thres)
            # xmax  = min(ori_width, coorx+thres)
            # ymax  = min(ori_height, coory+thres)
            # area  = current_channel[ymin:ymax, xmin:xmax]
            # score = np.sum(area)
            score = 1.
            #
            # for y in range(ymin, ymax, 1):
            #     for x in range(xmin, xmax, 1):
            #         xscore += current_channel[y][x] * x
            #         yscore += current_channel[y][x] * y
            # print('2---(x, y) == ', (xscore, yscore))

        keypoints.append([coorx, coory])
        pose_score += score

    return keypoints, pose_score / 17


def make_prn_inputs(keypoints, prn_boxs):
    '''
    根据传入的keypoints和boxs得到PRN网络的输出，注意此时keypoints和boxs的数值，都是相对于原始图片的，也即是没有经过任何缩放的
    :param keypoints: list of list， 每个元素是一个长度为17*2的列表，依次为 [x0, y0, x1, y1....]
    :param boxs: list of list or ndarray, 每个元素是一个长度为4的列表，即box的[xmin, ymin, xmax, ymax]，所有值都是相对于原始图像而言
    :return:
        prn_inputs:
            列表的列表， 其中每个列表都是[56, 36, 17]大小的ndarray
    '''
    boxs = prn_boxs.copy()
    prn_inputs = []
    in_thres = 0.21
    keypoints = np.asarray(keypoints)
    # print ('keypoints shape ', keypoints.shape)
    w = 36
    h = 56

    for b in list(boxs):
        b[2] = b[2] - b[0]
        b[3] = b[3] - b[1]
        single_prn_input = np.zeros((h, w, 17))
        for single_kps in list(keypoints):
            single_kps = np.reshape(np.asarray(single_kps), newshape=(17, -1))
            for c in range(num_keypoints):
                points_in_channel_c = single_kps[c, :]
                p_x = points_in_channel_c[0]
                p_y = points_in_channel_c[1]
                if p_x == 0 and p_y == 0:
                    continue

                # b是box，[xmin, ymin, w, h]
                # 下面的过程就和在训练pose-residual-net时生成训练数据是一样的。
                # 判断关键点是否在当前的box内，如果是，就根据缩放比例把weights_bbox对应的位置处表示为instance的值
                is_inside = p_x > b[0] - b[2] * in_thres and \
                            p_y > b[1] - b[3] * in_thres and \
                            p_x < b[0] + b[2] * (1.0 + in_thres) and \
                            p_y < b[1] + b[3] * (1.0 + in_thres)

                if is_inside:
                    x_scale = float(w) / math.ceil(b[2])
                    y_scale = float(h) / math.ceil(b[3])

                    x0 = int((p_x - b[0]) * x_scale)
                    y0 = int((p_y - b[1]) * y_scale)

                    if x0 >= w and y0 >= h:
                        x0 = w - 1
                        y0 = h - 1
                    elif x0 >= w:
                        x0 = w - 1
                    elif y0 >= h:
                        y0 = h - 1
                    elif x0 < 0 and y0 < 0:
                        x0 = 0
                        y0 = 0
                    elif x0 < 0:
                        x0 = 0
                    elif y0 < 0:
                        y0 = 0
                    single_prn_input[y0][x0][c] = 1

        prn_inputs.append(single_prn_input)
    # print ('boxs ', boxs)
    # print ('prn boxs', prn_boxs)
    return prn_inputs


def select_pred_boxs(loc_preds, cls_preds):

    # -------------------------(1) generate anchor-----------------------------------------#
    input_size = [tf.to_float(img_size), tf.to_float(img_size)]
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

def read_json(json_file):
    '''

    :param json_file:
    :return:
        img_ids: list, congtains img_id
        img_keypoints: list of list, contains every person's keypoint on image
        img_boxs: list of list, contains every person's box on image, [xmin, ymin, xmax, ymax]
    '''
    f = open(json_file, encoding='utf-8')
    labels = json.load(f)
    img_ids = []
    img_keypoints = []
    img_boxs = []
    img_names = []

    for label in labels:
        img_names.append(label['image_id'])
        img_ids.append(label['id'])
        kps = label['keypoint_annotations']
        boxs = label['human_annotations']
        kp = []
        box = []

        for key, value in kps.items():
            single_kp = value
            single_box = boxs[key]
            kp.append(single_kp)
            box.append(single_box)

        img_keypoints.append(kp)
        img_boxs.append(box)

    return img_names, img_keypoints, img_boxs, img_ids

def prepare_heatmaps(pred_heatmap, ori_height, ori_width):
    '''
    get pre_keypoints based on pred_heatmap
    :param pred_heatmap: keypoint subnet output. shape == [1, img_size//4, img_size//4, 17]
    :param ori_height: int, the original height of original img
    :param ori_width:  int, the original width of original img
    :return:
        pred_keypoints:
            ndarray, shape = [-1, 17, 2], 最后一维是两个坐标值，[x, y]， 均是相对于原图而言

    '''
    pred_keypoints = []
    factory = ori_height / pred_heatmap.shape[1]
    factorx = ori_width / pred_heatmap.shape[2]
    max_count = 0
    for i in range(1):
        current_pred_heatmap = pred_heatmap[i]
        for c in range(num_keypoints):
            current_channel = current_pred_heatmap[:, :, c]
            #-----------------find possible point location for single channel c---------#
            threshold = point_score
            nms_thres = point_nms_thres
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

            #--------save current channel possiable points cooridinat.-------------#
            current_channel_kps = []
            current_count       = 0
            for index in keep_index:
                coord = coordinate[index]
                current_channel_kps.append([coord[1]*factorx, coord[0]*factory])
                current_count += 1
            if max_count < current_count:
                max_count = current_count

            pred_keypoints.append(current_channel_kps)
    #-----transfer pred_keypoints to [-1, 17, 2]----#
    new_pred_keypoints = []
    for current_channel_kps in pred_keypoints:
        c_ll = len(current_channel_kps)
        for i in range(c_ll, max_count, 1):
            current_channel_kps.append([0., 0.,])

        new_pred_keypoints.append(current_channel_kps)
        # print (current_channel_kps)

    new_pred_keypoints = np.asarray(new_pred_keypoints)

    pred_keypoints = []
    for person in range(new_pred_keypoints.shape[1]):
        single_person_kps = new_pred_keypoints[:, person, :]
        single_person_kps = np.reshape(single_person_kps, newshape=(17, 2))
        # print (single_person_kps)
        pred_keypoints.append(single_person_kps)

    return np.asarray(pred_keypoints)

def translate(boxs, ori_height, ori_width):
    '''
    translate [ymin, xmin, ymax, xmax] into [xmin, ymin, xmax, ymax]
    :param boxs:
    :return:
    '''
    new_boxs = []
    factorx  = ori_width / img_size
    factory  = ori_height / img_size

    for box in boxs:
        new_box = [1, 2, 3, 4]
        new_box[0] = box[1] * factorx
        new_box[1] = box[0] * factory
        new_box[2] = box[3] * factorx
        new_box[3] = box[2] * factory
        new_boxs.append(new_box)

    return np.asarray(new_boxs)

def get_yolo_boxs(yolo_output, ori_shape, resized_shape):
    boxes = convert_box_coordinates(yolo_output)
    filtered_boxes = non_max_suppression(boxes, confidence_threshold=0.5, iou_threshold=0.4)

    # classes
    names = {}
    with open('yolo_v3/utils/coco_classes.txt') as f:
        class_names = f.readlines()
        for id, name in enumerate(class_names):
            names[id] = name

    height_ratio = ori_shape[0] / resized_shape[0]
    width_ratio = ori_shape[1] / resized_shape[1]
    ratio = (width_ratio, height_ratio)

    yolo_boxs = []

    for object_class, box_coords_and_prob in filtered_boxes.items():
        if str(names[object_class])[:-1] != 'person':
            continue
        for box_coord, object_prob in box_coords_and_prob:
            box_coord = box_coord.reshape(2, 2) * ratio
            box_coord = box_coord.reshape(-1)

            xmin = int(box_coord[0])
            ymin = int(box_coord[1])
            xmax = int(box_coord[2])
            ymax = int(box_coord[3])

            yolo_boxs.append([xmin, ymin, xmax, ymax])

    return yolo_boxs

multi_pose_net_eval()