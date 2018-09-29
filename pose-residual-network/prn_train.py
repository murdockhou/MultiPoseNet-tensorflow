# encoding: utf-8
'''
@author: shiwei hou
@contact: murdockhou@gmail.com
@software: PyCharm
@file: prn_train.py.py
@time: 18-9-28 上午9:30
'''

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import os, json, cv2, time

from datetime import datetime

from src.PRN import  PRN
from src.reader import PRN_READER

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_integer('batch_size', 21, '')
tf.flags.DEFINE_float('learning_rate', 1e-4, '')
tf.flags.DEFINE_integer('height', 56, '')
tf.flags.DEFINE_integer('width', 36, '')
tf.flags.DEFINE_integer('channels', 14, '')
tf.flags.DEFINE_boolean('is_training', True, '')
tf.flags.DEFINE_string('tfrecord_file', '/media/ulsee/E/pose_residual_net_tfrecord/ai_train2017.tfrecord', '')
tf.flags.DEFINE_string('checkpoint_path', '/media/ulsee/D/PRN', 'path to save training model')
tf.flags.DEFINE_string('finetuning', None,
                       'folder of saved model that you wish to continue training or testing(e.g. 20180828-1803/model.ckpt-xxx), default:None')

def BCEloss(labels, inputs):
    return tf.reduce_mean(
        -(tf.multiply(labels, tf.log(inputs)) +
          tf.multiply((1-labels), tf.log(1-inputs)))
    )

def prn_train():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    #-------------------define where checkpoint path is-------------------------#
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
    print ('checkpoints_dir == {}'.format(checkpoints_dir))

    #------------------------------define Graph --------------------------------#
    tf.reset_default_graph()
    graph = tf.Graph()
    with graph.as_default():
        #-----------------------------reader------------------------------------#
        reader = PRN_READER(batch_size=FLAGS.batch_size, height=FLAGS.height, width=FLAGS.width, channels=FLAGS.channels,
                            tfrecord_file=FLAGS.tfrecord_file)
        inputs, label = reader.feed()
        # print (inputs.get_shape())
        # print (label.get_shape())
        #----------------------------PRN Model----------------------------------#
        model = PRN(inputs=inputs, output_node=FLAGS.height*FLAGS.width*FLAGS.channels, is_training=FLAGS.is_training)
        out   = model.forward()
        #------------------------------Saver------------------------------------#
        saver = tf.train.Saver(max_to_keep=10)
        #------------------------------Loss-------------------------------------#
        # loss  = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=out)) / tf.to_float(FLAGS.batch_size)
        loss  = BCEloss(labels=label, inputs=out)
        # print (loss.get_shape())
        #---------------------------lr and gradient-----------------------------#
        global_step   = tf.Variable(0, trainable=False)
        learning_rate = tf.to_float(FLAGS.learning_rate)
        opt           = tf.train.AdamOptimizer(learning_rate, epsilon=1e-5)
        grads         = opt.compute_gradients(loss)
        apply_grad_op = opt.apply_gradients(grads, global_step=global_step)
        #-----------------------------tf summary--------------------------------#
        gt_label   = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, FLAGS.height, FLAGS.width, FLAGS.channels))
        pred_label = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, FLAGS.height, FLAGS.width, FLAGS.channels))
        tf.summary.scalar('lr', learning_rate)
        tf.summary.scalar('loss', loss)
        # tf.summary.image('label', tf.reshape(tf.transpose(
        #     gt_label, [3, 0, 1, 2])[6], shape=(-1, FLAGS.height, FLAGS.width, 1)), max_outputs=4)
        # tf.summary.image('pred', tf.reshape(tf.transpose(
        #     pred_label, [3, 0, 1, 2])[6], shape=(-1, FLAGS.height, FLAGS.width, 1)), max_outputs=4)
        summary_op     = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(checkpoints_dir, graph)
        #--------------------------------init------------------------------------#
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        config  = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        #--------------------------------train------------------------------------#
        with tf.Session(graph=graph, config=config) as sess:
            sess.run(init_op)
            coord   = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            step    = 0
            s_time  = time.time()
            try:
                while not coord.should_stop() and step < 100000:
                    net_x, y = sess.run([inputs, label])

                    _, net_loss, lr, merge_op = sess.run(
                        [apply_grad_op, loss, learning_rate, summary_op],
                        feed_dict={gt_label:y, pred_label:net_x}
                    )

                    summary_writer.add_summary(merge_op, step)
                    summary_writer.flush()

                    if (step + 1) % 10000 == 0:
                        save_path = saver.save(sess, checkpoints_dir + '/model.ckpt', global_step=step)
                        print ('Model saved in {}'.format(save_path))
                    if (step + 1) % 100 == 0:
                        cur_time = time.time()
                        print ('step {}: loss = {}, time spend = {}'.format(step, net_loss, cur_time-s_time))
                        s_time = cur_time

                    step += 1
                    # break

            except KeyboardInterrupt:
                print ('Interrupted')
                coord.request_stop()
            except Exception as e:
                coord.request_stop(e)
            finally:
                save_path = saver.save(sess, checkpoints_dir + '/model.ckpt', global_step=step)
                print('Model saved in {}'.format(save_path))
                coord.request_stop()
                coord.join(threads)


if __name__ == '__main__':
    prn_train()





