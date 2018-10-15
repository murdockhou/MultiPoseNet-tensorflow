# encoding: utf-8
'''
@author: shiwei hou
@contact: murdockhou@gmail.com
@software: PyCharm
@file: person_detect_test.py
@time: 18-10-9 下午5:14
'''

# encoding: utf-8
'''
@author: shiwei hou
@contact: murdockhou@gmail.com
@software: PyCharm
@file: test_model.py
@time: 18-9-21 下午1:31
'''

import tensorflow as tf
import numpy as np
import json, cv2, os
import logging

from src.RetinaNet import RetinaNet
from anchor.anchor_generator import create_retinanet_anchors
from anchor.box_coder import FasterRCNNBoxCoder


FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('model', '/media/ulsee/D/retinanet/20180925-2050/model.ckpt-41999',
                       'model path you want to test, e.g,. (/media/ulsee/D/retinanet/20180920-1337/model.ckpt-xxxx')
tf.flags.DEFINE_string('img_path', '/media/ulsee/E/datasets/test',
                       'img path to test model')
tf.flags.DEFINE_string('save_path', '/media/ulsee/E/retinanet/test',
                       'model test result to save')
tf.flags.DEFINE_integer(name='batch_size', default=1, help='train batch size number')
tf.flags.DEFINE_integer(name='img_size', default=600, help='net input size')
tf.flags.DEFINE_boolean('is_single_channel', False, 'define the net cls_pred is single channel or not.')

def draw_boxs(img, boxs, scores):

    for i in range(len(scores)):
        box = boxs[i]
        cv2.rectangle(img, (box[1], box[0]), (box[3], box[2]), (255 - 10*i,0,0), 1)
        cv2.putText(img, 'person: ' + str(scores[i]), (box[1], box[0]), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 1)
    return img

def net_():
    if not os.path.exists(FLAGS.save_path):
        os.makedirs(FLAGS.save_path)

    graph = tf.Graph()
    with graph.as_default():
        inputs = tf.placeholder(dtype=tf.float32, shape=[FLAGS.batch_size, FLAGS.img_size, FLAGS.img_size, 3], name='inputs')
        if FLAGS.is_single_channel:
            model = RetinaNet(inputs=inputs, batch_size=FLAGS.batch_size, num_classes=1)
        else:
            model = RetinaNet(inputs=inputs, batch_size=FLAGS.batch_size, num_classes=2)
        cls_preds, loc_preds = model.class_predictions, model.loc_predictions

        #-------------------------------generate anchor----------------------------------------#
        input_size = [tf.to_float(FLAGS.img_size), tf.to_float(FLAGS.img_size)]
        feature_map_list = [(tf.ceil(tf.multiply(input_size[0], 1 / pow(2., i + 3))),
                             tf.ceil(tf.multiply(input_size[1], 1 / pow(2., i + 3))))
                            for i in range(5)]
        anchor_generator = create_retinanet_anchors()
        anchor = anchor_generator.generate(input_size, feature_map_list)

        # -------------------------------decode loc_pred---------------------------------------#
        current_loc_pred = loc_preds[0]
        # 根据anchor将网络的loc输出解码，表示为[ymin, xmin, ymax, xmax]
        current_box_list = FasterRCNNBoxCoder().decode(current_loc_pred, anchor.get())
        current_decoded_loc_pred = current_box_list.get()
        # -------------------------------------NMS--------------------------------------------#
        if FLAGS.is_single_channel:
            box_score = tf.nn.sigmoid(cls_preds[0])
            box_score = box_score[:, 0]
        else:
            box_score = tf.nn.softmax(cls_preds[0])
            box_score = box_score[:, 1]
        top_k_score, top_k_indices = tf.nn.top_k(box_score, k=60)
        top_k_boxs = tf.gather(current_decoded_loc_pred, top_k_indices)
        nms_indices = tf.image.non_max_suppression(boxes=top_k_boxs, scores=top_k_score, max_output_size=5, iou_threshold=0.5)
        final_boxs = tf.gather(top_k_boxs, nms_indices)
        final_scores = tf.gather(top_k_score, nms_indices)
        #-------------------------------draw image with boxs-------------------------------------#
        _boxs = tf.expand_dims(final_boxs / tf.to_float(FLAGS.img_size), axis=0)
        _img = inputs
        _img_with_box = tf.image.draw_bounding_boxes(images=_img, boxes=_boxs)
        #----------------------------------------------------------------------------------------#

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(init_op)
            saver.restore(sess, FLAGS.model)
            logging.info('model restore successfully.')

            #----------------load img-----------------#
            img_num = 0
            for img in os.listdir(FLAGS.img_path):
                img_ori = cv2.imread(os.path.join(FLAGS.img_path, img), cv2.IMREAD_COLOR)
                img_copy = img_ori.copy()
                #------------deal img-----------------#
                height, width, channels = img_copy.shape
                if height > width:
                    img_padding = cv2.copyMakeBorder(img_copy, 0, 0, 0, height - width, cv2.BORDER_CONSTANT,
                                                     value=(0, 0, 0))
                else:
                    img_padding = cv2.copyMakeBorder(img_copy, 0, width - height, 0, 0, cv2.BORDER_CONSTANT,
                                                     value=(0, 0, 0))

                img_input = cv2.resize(img_padding, (FLAGS.img_size, FLAGS.img_size),
                                       interpolation=cv2.INTER_NEAREST)
                # boxs, [n, 4], n = [ymin, xmin, ymax, xmax]
                classes, locations, boxs, scores, img_boxs = sess.run([cls_preds, loc_preds, final_boxs, final_scores, _img_with_box],
                                                            feed_dict={inputs:[img_input]})

                #--------------------scale------------------#
                factorx = img_padding.shape[1] / img_input.shape[1]
                factory = img_padding.shape[0] / img_input.shape[0]
                boxs[:,0] = boxs[:,0] * factory
                boxs[:,2] = boxs[:,2] * factory
                boxs[:,1] = boxs[:,1] * factorx
                boxs[:,3] = boxs[:,3] * factorx
                #-------------------------------------------#
                img_save = draw_boxs(img_ori, boxs, scores)
                cv2.imwrite(os.path.join(FLAGS.save_path, img), img_save)
                cv2.imwrite(os.path.join(FLAGS.save_path, 'tf' + img), img_boxs[0])
                img_num += 1
                logging.info('Testing imgs ... {}'.format(img_num))

                if img_num > 100:
                    break


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    net_()

