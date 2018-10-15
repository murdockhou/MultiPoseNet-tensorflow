# encoding: utf-8
'''
@author: shiwei hou
@contact: murdockhou@gmail.com
@software: PyCharm
@file: keypoint_train.py
@time: 18-9-28 下午12:13
'''

import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile
from datetime import datetime
import os, time, cv2
import numpy as np

from src.backbone import BackBone
from src.model import Keypoint_Subnet
from src.get_heatmap import get_heatmap
from src.reader import Keypoint_Reader
from src.json_read import  load_json, load_coco_json
from src.img_pre_processing import img_pre_processing



FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer('train_nums', 118280, 'train data nums, default: cocotrain2017--118280')
tf.flags.DEFINE_integer('epochs', 10, 'train epochs')
tf.flags.DEFINE_integer('batch_size', 4, 'train batch size number')
tf.flags.DEFINE_integer('img_size', 480, 'net input size')
tf.flags.DEFINE_float('learning_rate', 1e-4, 'trian lr')
tf.flags.DEFINE_float('decay_rate', 0.95, 'lr decay rate')
tf.flags.DEFINE_integer('decay_steps', 10000, 'lr decay steps')
tf.flags.DEFINE_integer('max_to_keep', 10, 'num of models to saved')
tf.flags.DEFINE_integer('num_keypoints', 17, 'number of keypoints to detect')
tf.flags.DEFINE_string('pretrained_resnet', 'pre_trained/resnet_v2_50.ckpt',
                       'resnet_v2_50 pretrained model')
tf.flags.DEFINE_boolean('is_training', True, '')
tf.flags.DEFINE_string('checkpoint_path', '/media/ulsee/D/keypoint_subnet', 'path to save training model')
tf.flags.DEFINE_string('tfrecord_file', '/media/ulsee/E/keypoint_subnet_tfrecord/coco_train2017.tfrecord', '')
tf.flags.DEFINE_string('json_file', '/media/ulsee/E/keypoint_subnet_tfrecord/coco_train2017.json',
                       '')
tf.flags.DEFINE_string('finetuning', None,
                       'folder of saved model that you wish to continue training or testing(e.g. 20180828-1803/model.ckpt-xxx), default:None')
tf.flags.DEFINE_boolean('change_dataset', False,
                        'if change dataset from ai_challenger to coco, the num_keypoints will be changed. If so, when we finetunnig, need to '
                        'specify do not restore the last output layer var.')

def keypoint_train():
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
    #-----------------------------load json--------------------------------------#
    imgid_keypoints_dict = load_json(FLAGS.json_file)
    # ------------------------------define Graph --------------------------------#
    # tf.reset_default_graph()
    graph = tf.Graph()
    with graph.as_default():
        #------------------------get backbone net--------------------------------#
        backbone = BackBone(img_size=FLAGS.img_size, batch_size=FLAGS.batch_size, is_training=FLAGS.is_training)
        fpn, _   = backbone.build_fpn_feature()
        #---------------------------keypoint net---------------------------------#
        keypoint_net = Keypoint_Subnet(inputs=backbone.input_imgs, img_size=backbone.img_size, fpn=fpn,
                                       batch_size=backbone.batch_size, num_classes=FLAGS.num_keypoints)
        total_loss, net_loss, pre_heat = keypoint_net.net_loss()
        #-------------------------------reader-----------------------------------#
        reader = Keypoint_Reader(tfrecord_file=FLAGS.tfrecord_file, batch_size=FLAGS.batch_size, img_size=FLAGS.img_size, epochs=FLAGS.epochs)
        img_batch, img_id_batch, img_height_batch, img_width_batch = reader.feed()
        #-----------------------------learning rate------------------------------#
        global_step   = tf.Variable(0)
        learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step=global_step,
                                                   decay_steps=FLAGS.decay_steps, decay_rate=FLAGS.decay_rate,
                                                   staircase=True)
        opt               = tf.train.AdamOptimizer(learning_rate, epsilon=1e-5)
        # grads             = opt.compute_gradients(total_loss)
        # apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

        # MOVING_AVERAGE_DECAY  = 0.99
        # variable_averages     = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
        # variable_to_average   = (tf.trainable_variables() + tf.moving_average_variables())
        # variables_averages_op = variable_averages.apply(variable_to_average)

        update_ops   = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = opt.minimize(total_loss, global_step=global_step)

        #--------------------------------saver-----------------------------------#
        res50_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='resnet_v2_50')
        restore_res50  = tf.train.Saver(var_list=res50_var_list)

        fpn_var_list             = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='build_fpn_feature')
        keypoint_subnet_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='keypoint_subnet')
        output_name              = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='keypoint_subnet.output')

        var_list        = tf.trainable_variables()
        global_list     = tf.global_variables()
        bn_moving_vars  = [g for g in global_list if 'moving_mean' in g.name]
        bn_moving_vars += [g for g in global_list if 'moving_variance' in g.name]
        var_list       += bn_moving_vars

        if FLAGS.change_dataset:
            for node in output_name:
                var_list.remove(node)

        if FLAGS.finetuning is not None:
            restore_finetuning = tf.train.Saver(var_list=var_list)

        saver       = tf.train.Saver(var_list=var_list, max_to_keep=20)
        saver_alter = tf.train.Saver(max_to_keep=5)

        #---------------------------------control sigma for heatmap-------------------------------#
        start_gussian_sigma    = 10.0
        end_gussian_sigma      = 2.5
        start_decay_sigma_step = 10000
        decay_steps            = 50000
        # gussian sigma will decay when global_step > start_decay_sigma_step
        gussian_sigma = tf.where(
            tf.greater(global_step, start_decay_sigma_step),
            tf.train.polynomial_decay(start_gussian_sigma,
                                      tf.cast(global_step, tf.int32) - start_decay_sigma_step,
                                      decay_steps,
                                      end_gussian_sigma,
                                      power=1.0),
            start_gussian_sigma
        )
        # --------------------------------init------------------------------------#
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        config  = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        #--------------------------------tf summary--------------------------------#
        img_id_batch_placeholder = tf.placeholder(tf.string, shape=[FLAGS.batch_size,])
        tf.summary.text('img_ids', img_id_batch_placeholder)
        tf.summary.scalar('total_loss', total_loss)
        tf.summary.scalar('net_loss', net_loss)
        tf.summary.image('gt_right_shoulder', tf.reshape(tf.transpose(
            keypoint_net.input_heats, [3, 0, 1, 2])[6], shape=[-1, FLAGS.img_size // 4, FLAGS.img_size // 4, 1]), max_outputs=2)
        tf.summary.image('ori_image', backbone.input_imgs, max_outputs=2)
        # tf.summary.image('gt_left_shoulder', tf.reshape(tf.transpose(
        #     keypoint_net.input_heats, [3, 0, 1, 2])[5], shape=[-1, FLAGS.img_size // 4, FLAGS.img_size // 4, 1]),max_outputs=2)
        tf.summary.image('pred_right_shoulder', tf.reshape(tf.transpose(
            pre_heat, [3, 0, 1, 2])[6], shape=[-1, FLAGS.img_size // 4, FLAGS.img_size // 4, 1]), max_outputs=2)
        tf.summary.image('gt_heatmap', tf.reduce_sum(keypoint_net.input_heats, axis=3, keepdims=True), max_outputs=2)
        tf.summary.image('pred_heatmap', tf.reduce_sum(pre_heat, axis=3, keepdims=True), max_outputs=2)
        tf.summary.scalar('lr', learning_rate)
        summary_op     = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(checkpoints_dir, graph)
        # --------------------------------train------------------------------------#
        with tf.Session(graph=graph, config=config) as sess:
            sess.run(init_op)
            coord   = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            step    = 0

            if FLAGS.finetuning is not None:
                restore_finetuning.restore(sess, checkpoints_dir)
                print ('Successfully load pre_trained keypoint_subnet model.')
                # step = int(checkpoints_dir.split('/')[-1].split('.')[-1].split('-')[-1])
                print ('Global_step == {}, Step == {}'.format(sess.run(global_step), step))
                # step = sess.run(global_step)
                # -- bn layer: resnet_v2_50/block1/unit_1/bottleneck_v2/conv1/BatchNorm/ ---#
                # gamma = graph.get_tensor_by_name(name='resnet_v2_50/block4/unit_3/bottleneck_v2/conv2/BatchNorm/gamma:0')
                # beta = graph.get_tensor_by_name(name='resnet_v2_50/block4/unit_3/bottleneck_v2/conv2/BatchNorm/beta:0')
                # print('finetuning gamma = ', sess.run(gamma)[:50])
                # print('beta = ', sess.run(beta)[:50])

            else:
                restore_res50.restore(sess, FLAGS.pretrained_resnet)
                print ('Successfully load pre_trained resnet_v2_50 model')
                # -- bn layer: resnet_v2_50/block1/unit_1/bottleneck_v2/conv1/BatchNorm/ ---#
                # gamma = graph.get_tensor_by_name(
                #     name='resnet_v2_50/block1/unit_1/bottleneck_v2/conv1/BatchNorm/gamma:0')
                # beta = graph.get_tensor_by_name(name='resnet_v2_50/block1/unit_1/bottleneck_v2/conv1/BatchNorm/beta:0')
                # print('no finetuning gamma = ', sess.run(gamma)[:50])
                # print('beta = ', sess.run(beta)[:50])

            start_time = time.time()
            try:
                while not coord.should_stop():
                    imgs, imgs_id, imgs_height, imgs_width, g_sigma = sess.run([img_batch, img_id_batch, img_height_batch, img_width_batch, gussian_sigma])

                    gt_heatmaps = get_heatmap(label_dict=imgid_keypoints_dict, img_ids=imgs_id, img_heights=imgs_height,
                                              img_widths=imgs_width, img_resize=FLAGS.img_size, num_keypoints=FLAGS.num_keypoints,
                                              sigma=g_sigma)

                    imgs, gt_heatmaps = img_pre_processing(imgs, gt_heatmaps)

                    _, loss_all, net_out_loss, pre_heats, lr, merge_op = sess.run(
                        [train_op, total_loss, net_loss, pre_heat, learning_rate, summary_op],
                        feed_dict={backbone.input_imgs:imgs,
                                   keypoint_net.input_heats:gt_heatmaps,
                                   img_id_batch_placeholder:imgs_id}
                    )

                    summary_writer.add_summary(merge_op, step)
                    summary_writer.flush()

                    if (step + 1) % 10 == 0:
                        cur_time = time.time()
                        print ('-------------------Step %d:-------------------' % step)
                        print ('total_loss = {}, out_put_loss = {}, lr = {}, sigma = {}, time spend = {}'
                                     .format(loss_all, net_out_loss, lr, g_sigma, cur_time-start_time))
                        start_time = cur_time

                        # # -- bn layer: resnet_v2_50/block1/unit_1/bottleneck_v2/conv1/BatchNorm/ ---#
                        # gamma = graph.get_tensor_by_name(
                        #     name='resnet_v2_50/block1/unit_1/bottleneck_v2/conv1/BatchNorm/gamma:0')
                        # beta = graph.get_tensor_by_name(
                        #     name='resnet_v2_50/block1/unit_1/bottleneck_v2/conv1/BatchNorm/beta:0')
                        # print('no finetuning gamma = ', sess.run(gamma)[:50])
                        # print('beta = ', sess.run(beta)[:50])
                        # print (sess.run(bn_moving_vars[0]))
                        # input_graph_def = tf.get_default_graph().as_graph_def()
                        # output_graph_def = graph_util.convert_variables_to_constants(sess, input_graph_def,
                        #                                                              'keypoint_subnet/output/biases'.split(','))
                        # model_f = tf.gfile.FastGFile('model.pb', 'wb')
                        # model_f.write(output_graph_def.SerializeToString())
                        # break
                    if (step + 1) % 5000 == 0:
                        save_path = saver.save(sess, checkpoints_dir + '/model.ckpt', global_step=step)
                        print ('Model saved in file: {}'.format(save_path))
                        save_path_alter = saver_alter.save(sess, checkpoints_dir+'/model_alter.ckpt', global_step=step)

                    step += 1


            except KeyboardInterrupt:
                print ('Interrupted, current step == {}'.format(step))
                coord.request_stop()

            except Exception as e:
                coord.request_stop(e)

            finally:
                save_path = saver.save(sess, checkpoints_dir + "/model.ckpt", global_step=step)
                print ("Model saved in file: {}" .format(save_path))
                save_path_alter = saver_alter.save(sess, checkpoints_dir + '/model_alter.ckpt', global_step=step)
                print ('Current step = {}'.format(step))
                # When done, ask the threads to stop.
                coord.request_stop()
                coord.join(threads)


if __name__ == '__main__':
    keypoint_train()









