# encoding: utf-8
'''
@author: shiwei hou
@contact: murdockhou@gmail.com
@software: PyCharm
@file: person_detect_train.py
@time: 18-9-28 下午2:47
'''

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import matplotlib.pyplot as plt
import os, json, cv2, time
from datetime import datetime

import  sys
sys.path.append('../')

from src.reader import Box_Reader
from src.get_loss import get_loss
from src.draw_box_with_image import get_gt_boxs_with_img, get_pred_boxs_with_img

from src.retinanet import RetinaNet
from keypoint_subnet.src.backbone import  BackBone

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer('train_nums', 118280, 'train data nums, default: cocotrain2017--118280')
tf.flags.DEFINE_integer('epochs', 10, 'train epochs')
tf.flags.DEFINE_integer('num_classes', 1, '')
tf.flags.DEFINE_integer('batch_size', 3, 'train batch size number')
tf.flags.DEFINE_integer('img_size', 480, 'net input size')
tf.flags.DEFINE_float('learning_rate', 5e-5, 'trian lr')
tf.flags.DEFINE_float('decay_rate', 0.9, 'lr decay rate')
tf.flags.DEFINE_integer('decay_steps', 10000, 'lr decay steps')
tf.flags.DEFINE_string('pretrained_resnet', '/media/ulsee/D/keypoint_subnet/model.ckpt-64999/model.ckpt-129999',
                       'keypoint subnet pretrained model')
tf.flags.DEFINE_boolean('is_training', True, '')
tf.flags.DEFINE_string('checkpoint_path', '/media/ulsee/D/retinanet', 'path to save training model')
tf.flags.DEFINE_string('tfrecord_file', '/media/ulsee/E/person_subnet_tfrecord/ai-instance-bbox.tfrecord', '')
tf.flags.DEFINE_string('finetuning',None,
                    'folder of saved model that you wish to continue training or testing(e.g. 20180828-1803/model.ckpt-xxx), default: None')

def person_detect_train():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # -------------------define where checkpoint path is-------------------------#
    current_time = datetime.now().strftime('%Y%m%d-%H%M')
    if FLAGS.finetuning is None:
        checkpoints_dir = os.path.join(FLAGS.checkpoint_path, current_time)
        if not os.path.exists(checkpoints_dir):
            try:
                os.makedirs(checkpoints_dir)
            except:
                pass
    else:
        checkpoints_dir = os.path.join(FLAGS.checkpoint_path, FLAGS.finetuning)
    print('checkpoints_dir == {}'.format(checkpoints_dir))

    # ------------------------------define Graph --------------------------------#
    tf.reset_default_graph()
    graph = tf.Graph()
    with graph.as_default():
        #-----------------------------tf.placeholder-----------------------------#
        gt_boxs_placeholder = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, 30, 4])
        gt_labels_placeholder = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, 30,])
        #-------------------------------reader-----------------------------------#
        reader = Box_Reader(tfrecord_file=FLAGS.tfrecord_file, img_size=FLAGS.img_size, batch_size=FLAGS.batch_size, epochs=FLAGS.epochs)
        img_batch, img_height_batch, img_width_batch, gt_boxs, gt_labels = reader.feed()
        #--------------------------------net-------------------------------------#
        backbone = BackBone(img_size=FLAGS.img_size, batch_size=FLAGS.batch_size, is_training=FLAGS.is_training)
        fpn, _   = backbone.build_fpn_feature()
        net      = RetinaNet(fpn=fpn, feature_map_dict=_, batch_size=backbone.batch_size,
                             num_classes=FLAGS.num_classes+1, is_training=FLAGS.is_training)
        loc_pred, cls_pred = net.forward()
        #---------------------------------loss-----------------------------------#
        loss, decoded_loc_pred = get_loss(img_size=FLAGS.img_size, batch_size=FLAGS.batch_size,
                                          gt_boxes=gt_boxs_placeholder, loc_pred=loc_pred,
                                          gt_labels=gt_labels_placeholder, cls_pred=cls_pred,
                                          num_classes=FLAGS.num_classes, is_training=FLAGS.is_training)
        # -----------------------------learning rate-------------------------------#
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step=global_step,
                                                   decay_steps=FLAGS.decay_steps, decay_rate=FLAGS.decay_rate,
                                                   staircase=True)
        opt          = tf.train.AdamOptimizer(learning_rate, epsilon=1e-5)
        update_ops   = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = opt.minimize(loss, global_step=global_step)
        #--------------------------------saver-----------------------------------#
        res50_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='resnet_v2_50')
        restore_res50  = tf.train.Saver(var_list=res50_var_list)
        fpn_var_list   = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='build_fpn_feature')

        global_list    = tf.global_variables()
        bn_moving_vars = [g for g in global_list if 'moving_mean' in g.name]
        bn_moving_vars += [g for g in global_list if 'moving_variance' in g.name]
        restore_share  = tf.train.Saver(var_list=(res50_var_list+fpn_var_list+bn_moving_vars))

        var_list        = tf.trainable_variables()
        retina_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='retina_net')
        saver           = tf.train.Saver(var_list=(res50_var_list+fpn_var_list+bn_moving_vars+retina_var_list), max_to_keep=10)
        saver_alter     = tf.train.Saver(max_to_keep=5)

        #-------------------------------tf summary--------------------------------#
        gt_img_batch_with_box    = get_gt_boxs_with_img(imgs=backbone.input_imgs, gt_boxs=gt_boxs_placeholder, gt_labels=gt_labels_placeholder,
                                                        batch_size=FLAGS.batch_size, img_size=FLAGS.img_size)
        pred_img_batch_with_box  = get_pred_boxs_with_img(imgs=backbone.input_imgs, decoded_boxs=decoded_loc_pred, cls_pred=cls_pred,
                                                          batch_size=FLAGS.batch_size, img_size=FLAGS.img_size)
        gt_img_box_placeholder   = tf.placeholder(tf.float32,
                                                  shape=(FLAGS.batch_size, FLAGS.img_size, FLAGS.img_size, 3))
        pred_img_box_placeholder = tf.placeholder(tf.float32,
                                                  shape=(FLAGS.batch_size, FLAGS.img_size, FLAGS.img_size, 3))
        tf.summary.image('gt_bbox', gt_img_box_placeholder, max_outputs=2)
        tf.summary.image('Pre_bbox', pred_img_box_placeholder, max_outputs=2)
        tf.summary.scalar('lr', learning_rate)
        tf.summary.scalar('loss', loss)

        summary_op     = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(checkpoints_dir, graph)

        #----------------------------------init-----------------------------------#
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        config  = tf.ConfigProto()
        # config.gpu_options.per_process_gpu_memory_fraction = 0.7
        # sudo rm -f ~/.nv
        config.gpu_options.allow_growth = True
        step = 0
        #---------------------------------train------------------------------------#
        with tf.Session(graph=graph, config=config) as sess:
            sess.run(init_op)

            if FLAGS.finetuning is not None:
                saver.save(sess, checkpoints_dir)
                print('Successfully load finetuning model.')
                print('Global_step == {}, Step == {}'.format(sess.run(global_step), step))
                step = sess.run(global_step)

            else:
                restore_share.save(sess, FLAGS.pretrained_resnet)
                print ('Successfully load pre_trained model.')

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            start_time = time.time()
            try:
                while not coord.should_stop() and step < 400000:
                    imgs, heights, widths, boxes, labels = sess.run([img_batch, img_height_batch, img_width_batch, gt_boxs, gt_labels])

                    gt_img_box, pre_img_box, \
                    total_loss, box_pred_list, classes_pred, \
                    _, lr= sess.run(
                        [gt_img_batch_with_box, pred_img_batch_with_box,
                         loss, decoded_loc_pred, cls_pred,
                         train_op, learning_rate
                         ], feed_dict={
                            backbone.input_imgs: imgs,
                            gt_boxs_placeholder: boxes,
                            gt_labels_placeholder:labels
                        }
                    )
                    # cur_time = time.time()
                    # print ('sess run spend {}'.format(cur_time-pre_time))
                    # pre_time = cur_time

                    #-------------------summary------------------------#
                    # gt_img_box_placeholder: gt_img_box,
                    merge_op = sess.run(summary_op,feed_dict={pred_img_box_placeholder:pre_img_box,
                                                              gt_img_box_placeholder: gt_img_box})
                    summary_writer.add_summary(merge_op, step)
                    summary_writer.flush()

                    # cur_time = time.time()
                    # print('merge op spend {}'.format(cur_time - pre_time))
                    # pre_time = cur_time

                    if (step+1) % 10 == 0:
                        cur_time = time.time()
                        print('Step = {}, Total loss = {}, time spend = {}'.format(step, total_loss, cur_time-start_time))
                        start_time = cur_time

                    if (step+1) % 2000 == 0:
                        save_path = saver.save(sess, checkpoints_dir + '/model.ckpt', global_step=step)
                        print('Model saved in file: %s' % save_path)
                        save_path_alter = saver_alter.save(sess, checkpoints_dir + '/model_alter.ckpt',
                                                           global_step=step)

                    step += 1
                    # print (step)
                    # if step == 10:
                    #     break

            except KeyboardInterrupt:
                print ('Interrupted')
                coord.request_stop()

            except Exception as e:
                coord.request_stop(e)

            finally:
                save_path = saver.save(sess, checkpoints_dir + '/model.ckpt', global_step=step)
                print ('Model saved in file: %s' % save_path)
                save_path_alter = saver_alter.save(sess, checkpoints_dir + '/model_alter.ckpt', global_step=step)
                # When done, ask the threads to stop.
                coord.request_stop()
                coord.join(threads)




if __name__ == '__main__':
    person_detect_train()