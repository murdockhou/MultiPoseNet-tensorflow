import os
import math
import json
import argparse
import numpy as np
from tqdm import tqdm
from random import shuffle

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from utils.gaussian import gaussian, crop, gaussian_multi_input_mp

import tensorflow as tf
from pose_residual_network.src.PRN import PRN


def eval(checkpoint = '', json_file = '/media/ulsee/E/datasets/coco/annotations2017/person_keypoints_val2017.json'):

    ckpt = checkpoint
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    graph = tf.Graph()
    with graph.as_default():
        inputs = tf.placeholder(tf.float32, shape=(1, 56, 36 , 17), name='inputs')
        prn = PRN(inputs=inputs, output_node=1*56*36*17, is_training=False)
        prn_out = prn.forward()
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        saver = tf.train.Saver()

        with tf.Session(graph=graph) as sess:
            sess.run(init_op)
            saver.restore(sess, ckpt)
            print ('prn model restore successfully.')
            print('------------Evaulation Started------------')

            peak_results, bbox_results, coco = prepare(json_file)

            image_ids = []
            my_results = []
            n_kernel = 15

            w = int(18 * 2)
            h = int(28 * 2)
            in_thres = 0.21
            # tqdm, Python里用来控制显示的进度条，相当于循环
            for p in tqdm(peak_results):
                idx = p['image_id']
                image_ids.append(idx)

                peaks = p['peaks']
                # 找到当前图片所标注的所有的boxes，是一个列表的列表，[ [], [], ... ,[]]，每个列表值是原始coco标注信息里的box值[x, y, w, h]
                bboxes = [k['bbox'] for k in bbox_results if k['image_id'] == idx]

                if len(bboxes) == 0 or len(peaks) == 0:
                    continue

                # 构建网络的输入
                weights_bbox = np.zeros((len(bboxes), h, w, 4, 17))
                # 对这个图片上所有的关键点信息进行处理,注意peaks是有17个元素的列表，对应coco数据集标注的17个关键点，每个元素可以有多个关键点，表示多个人的同一个部位
                for joint_id, peak in enumerate(peaks):
                    # peak就是第几个channel上的所有关键点，也即是这个图片上所有的同一个类型的关键点信息，例如所有的鼻子、左肩、右肩等
                    for instance_id, instance in enumerate(peak):
                        # instance_id是当前channel上第几个点，instance是点，有四个值[x, y, 1, idx]
                        p_x = instance[0]
                        p_y = instance[1]

                        for bbox_id, b in enumerate(bboxes):
                            # bbox_id 表示第几个box，b是box，[xmin, ymin, w, h]
                            # 下面的过程就和在训练pose-residual-net时生成训练数据是一样的。
                            # 判断关键点是否在当前的box内，如果是，就根据缩放比例把weights_bbox对应的位置处表示为instance的值
                            # ?没有很看懂为什么weights_box维度是[ len(bboxes), h, w, 4, 17],感觉完全就可以是[ len(bboxes), h, w, 17]?
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

                                p = 1e-9

                                weights_bbox[bbox_id, y0, x0, :, joint_id] = [1, instance[2], instance[3], p]

                old_weights_bbox = np.copy(weights_bbox)

                for j in range(weights_bbox.shape[0]):
                    for t in range(17):
                        weights_bbox[j, :, :, 0, t] = gaussian(weights_bbox[j, :, :, 0, t])
                    # weights_bbox[j, :, :, 0, :]      = gaussian_multi_input_mp(weights_bbox[j, :, :, 0, :])

                # -------------------get output of prn net--------------------#
                output_bbox = []
                for j in range(weights_bbox.shape[0]):
                    inp = weights_bbox[j, :, :, 0, :]  # [h, w, 17]
                    output = sess.run(prn_out, feed_dict={inputs:[inp]})
                    temp = np.reshape(output, (56, 36, 17))
                    output_bbox.append(temp)

                # output_box: [len(bboxes), 56, 36, 17]
                output_bbox = np.array(output_bbox)


                keypoints_score = []

                for t in range(17):
                    indexes = np.argwhere(old_weights_bbox[:, :, :, 0, t] == 1)
                    keypoint = []
                    for i in indexes:

                        cr = crop(output_bbox[i[0], :, :, t], (i[1], i[2]), N=n_kernel)
                        score = np.sum(cr)

                        kp_id = old_weights_bbox[i[0], i[1], i[2], 2, t]
                        kp_score = old_weights_bbox[i[0], i[1], i[2], 1, t]
                        p_score = old_weights_bbox[i[0], i[1], i[2], 3, t]  ## ??
                        bbox_id = i[0]

                        score = kp_score * score

                        s = [kp_id, bbox_id, kp_score, score]

                        keypoint.append(s)
                    keypoints_score.append(keypoint)

                bbox_keypoints = np.zeros((weights_bbox.shape[0], 17, 3))
                bbox_ids = np.arange(len(bboxes)).tolist()

                # kp_id, bbox_id, kp_score, my_score
                for i in range(17):
                    joint_keypoints = keypoints_score[i]
                    if len(joint_keypoints) > 0:

                        kp_ids = list(set([x[0] for x in joint_keypoints]))

                        table = np.zeros((len(bbox_ids), len(kp_ids), 4))

                        for b_id, bbox in enumerate(bbox_ids):
                            for k_id, kp in enumerate(kp_ids):
                                own = [x for x in joint_keypoints if x[0] == kp and x[1] == bbox]

                                if len(own) > 0:
                                    table[bbox, k_id] = own[0]
                                else:
                                    table[bbox, k_id] = [0] * 4

                        for b_id, bbox in enumerate(bbox_ids):

                            row = np.argsort(-table[bbox, :, 3])

                            if table[bbox, row[0], 3] > 0:
                                for r in row:
                                    if table[bbox, r, 3] > 0:
                                        column = np.argsort(-table[:, r, 3])

                                        if bbox == column[0]:
                                            bbox_keypoints[bbox, i, :] = [x[:3] for x in peaks[i] if x[3] == table[bbox, r, 0]][0]
                                            break
                                        else:
                                            row2 = np.argsort(table[column[0], :, 3])
                                            if row2[0] == r:
                                                bbox_keypoints[bbox, i, :] = \
                                                [x[:3] for x in peaks[i] if x[3] == table[bbox, r, 0]][0]
                                                break
                    else:
                        for j in range(weights_bbox.shape[0]):
                            b = bboxes[j]
                            x_scale = float(w) / math.ceil(b[2])
                            y_scale = float(h) / math.ceil(b[3])

                            for t in range(17):
                                indexes = np.argwhere(old_weights_bbox[j, :, :, 0, t] == 1)
                                if len(indexes) == 0:
                                    max_index = np.argwhere(output_bbox[j, :, :, t] == np.max(output_bbox[j, :, :, t]))
                                    bbox_keypoints[j, t, :] = [max_index[0][1] / x_scale + b[0],
                                                               max_index[0][0] / y_scale + b[1], 0]

                my_keypoints = []

                for i in range(bbox_keypoints.shape[0]):
                    k = np.zeros(51)
                    k[0::3] = bbox_keypoints[i, :, 0]
                    k[1::3] = bbox_keypoints[i, :, 1]
                    k[2::3] = [2] * 17

                    pose_score = 0
                    count = 0
                    for f in range(17):
                        if bbox_keypoints[i, f, 0] != 0 and bbox_keypoints[i, f, 1] != 0:
                            count += 1
                        pose_score += bbox_keypoints[i, f, 2]
                    pose_score /= 17.0

                    my_keypoints.append(k)

                    image_data = {
                        'image_id': idx,
                        'bbox': bboxes[i],
                        'score': pose_score,
                        'category_id': 1,
                        'keypoints': k.tolist()
                    }
                    my_results.append(image_data)


        ann_filename = 'val2017_PRN_keypoint_results_prn.json'
        # write output
        json.dump(my_results, open(ann_filename, 'w'), indent=4)

        # load results in COCO evaluation tool
        coco_pred = coco.loadRes(ann_filename)

        # run COCO evaluation
        coco_eval = COCOeval(coco, coco_pred, 'keypoints')
        coco_eval.params.imgIds = image_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

    # os.remove(ann_filename)

def prepare(json_file):

    cocodir = json_file
    ann = json.load(open(cocodir))
    bbox_results = ann['annotations']

    coco = COCO(cocodir)
    img_ids = coco.getImgIds(catIds=[1])

    peak_results = []
    # peak_results 是一个列表，里面的每一个元素是一个字典，字典有三个key，分别是image_id, peaks, file_name. image_id 和 file_name就是coco数据集里图片的名字和ID
    # peaks，是一个列表，有17个元素，每个元素又是一个列表，每个元素又包含N个列表，这个列表有两种情况：
    # 1. 根据原有的关键点信息，当其v为大于0的时候（即可以在图片上标注，无论是否可见），就将四个值[x,y,v,idx]组成的列表当做这个列表的元素放进去，如果这个图片上标注了
    #    多个人，那么继续找到关键点，同样组成的四个值[x,y,1,idx]放进去
    #    其中x和y就是coco数据集标注的关键点的位置，v统一为1，idx指明这个关键点是第几个可以标注的关键点序号，从0开始
    # 2. 如果原有的关键点v为0，则该列表为空
    # 所以 peaks最终的内容就有可能为[ [], [], [[x, y, 1, 0]], [], [[x, y, 1, 1], [x,y,1,2], [x,y,1,3]], [[x, y, 1, 4]], [], ..., [[x, y, 1 idx]] ] 这种形式
    for i in img_ids:
        anns = coco.loadAnns(coco.getAnnIds(imgIds=i))
        # kps是图片上所有人的关键点信息，可能有多个人，也就是列表的列表形式，[ [keypoint1], [keypoints2] ]
        kps = [a['keypoints'] for a in anns]

        idx = 0

        ks = []
        for i in range(17):
            t = []
            for k in kps:
                x = k[0::3][i]
                y = k[1::3][i]
                v = k[2::3][i]

                if v > 0:
                    t.append([x, y, 1, idx])
                    idx += 1
            ks.append(t)
        image_id = anns[0]['image_id']
        peaks = ks

        element = {
            'image_id': image_id,
            'peaks': peaks,
            'file_name': coco.loadImgs(image_id)[0]['file_name']
        }

        peak_results.append(element)

    shuffle(peak_results)

    # temporary_peak_res，最终得到的结果就是去除掉前面peaks全为空的情况，即某张图片上一个关键点都没有的，给去掉，只保留起码有 >= test_keypoint_count的图片
    temporary_peak_res = []
    for p in peak_results:
        if (sum(1 for i in p['peaks'] if i != []) >= 0):
            temporary_peak_res.append(p)
    peak_results = temporary_peak_res

    return peak_results, bbox_results, coco


eval()