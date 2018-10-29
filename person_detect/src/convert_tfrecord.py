# encoding: utf-8
'''
@author: shiwei hou
@contact: murdockhou@gmail.com
@software: PyCharm
@file: convert_tfrecord.py.py
@time: 18-9-28 下午6:55
''' 
import tensorflow as tf
import cv2, os, json
import numpy as np
import matplotlib.pyplot as plt

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('json_file', '/media/ulsee/E/datasets/coco/annotations2017/coco-instance-imgid-bbox.json', '')
tf.flags.DEFINE_string('img_path', '/media/ulsee/E/datasets/coco/cocotrain2017', 'image dataset path need to convert to tfrecord')
tf.flags.DEFINE_string('tfrecord_file', '/media/ulsee/E/person_subnet_tfrecord/coco-instance-with-ids.tfrecord', 'tfrecord file')

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

def _process_one_image(img_file, bboxes, person_id=1):
    '''

    :param img_file: the img file that will be read and processing
    :param bboxes: a list, contains box and crossponding label, format is [xmin, ymin, w, h, category_id]*n, n is the number of annotationed person
    :param person_id: the category_id that person is
    :return:
    img_data: binary image file that reading from tf.gfile.FastGFile
    img_shape: [height, widht, channels] of img
    bboxs: a list, [ymin, xmin, ymax, xmax] * n
    labels: a list, [person_id,], size = n
    '''

    # read img data
    img_data = tf.gfile.FastGFile(img_file, 'rb').read()
    img_shape = cv2.imread(img_file).shape

    # deal with bboxes
    bboxs = []
    labels = []
    box_num = len(bboxes) // 5

    for i in range(box_num):
        if bboxes[i*5+4] != person_id:
            continue
        box = bboxes[i*5:i*5+4]
        label = bboxes[i*5+4]
        box[2] += box[0]
        box[3] += box[1]
        #----convert box format [xmin, ymin, xmax, ymax] to [ymin, xmin, ymax, xmax]-------#
        tmp = box[0]
        box[0] = box[1]
        box[1] = tmp
        tmp = box[2]
        box[2] = box[3]
        box[3] = tmp
        #-----------------------------------------------------------------------------------#
        bboxs += box
        labels.append(label)

    return img_data, img_shape, bboxs, labels

def convert_to_tfrecord(json_file, tfrecord_file):
    '''
    especially reading coco-json file

    :param json_file: prepared json_file that contains coco dataset person annotations, the format is a map, which key is img_name without suffix, and value is
    a list contains person_num * 5 elements, the each five elements is like [xmin, ymin, w, h, category_id].
    :param tfrecord_file: the tfrecord file that we save
    :return:
    '''

    tfrecord_dir = os.path.dirname(tfrecord_file)
    if not os.path.exists(tfrecord_dir):
        os.makedirs(tfrecord_dir)

    writer = tf.python_io.TFRecordWriter(tfrecord_file)
    f = open(json_file, encoding='utf-8')
    labels = json.load(f)

    total_img_nums = len(labels)
    count = 0
    count_zero = 0
    for key, value in labels.items():
        img_name = key + '.jpg'
        img_data, shape, bboxs, labels = _process_one_image(os.path.join(FLAGS.img_path, img_name), value)
        if not bboxs:
            count_zero += 1
            continue

        # ----if len(bboxs)//4 < n (set n = 20), add zeros to make len(bboxs)//4 == n------------#
        n = 30
        if len(bboxs) < n * 4:
            last = n * 4 - len(bboxs)
            bboxs += list(np.zeros(last, dtype=np.float32))
            labels += list(np.zeros(last // 4, dtype=np.int32))
        else:
            bboxs = bboxs[:n * 4]
            labels = labels[:n]
        # ----------------------------------------------------------------------------------------#

        img_format = b'JPEG'
        example = tf.train.Example(features=tf.train.Features(
            feature = {
                'image':_bytes_feature(img_data),
                'id':_string_feature(bytes(key, encoding='utf-8')),
                'height':_int64_feature(shape[0]),
                'width':_int64_feature(shape[1]),
                'format':_bytes_feature(img_format),
                'channel':_int64_feature(shape[2]),
                'boxes':_float_feature(bboxs), # [xmin, ymin, xmax, ymax] * 30
                'labels':_int64_feature(labels)
            }
        ))
        writer.write(example.SerializeToString())
        count += 1

        # if count == 5:
        #     break

        if count % 1000 == 0:
            print ('Processing {}/{}'.format(count, total_img_nums))
    print ('No human box imgs nums {}/{}'.format(count_zero, total_img_nums))
    print('Converting tfrecord done.')
    writer.close()

def convert_ai_challenger_tfrecord(tfrecord_file, json_file = '/media/ulsee/E/datasets/ai_challenger_keypoint_train_20170909/keypoint_train_annotations_20170909.json'):
    f = open(json_file, encoding='utf-8')
    labels = json.load(f)
    img_path = '/media/ulsee/E/datasets/ai_challenger_keypoint_train_20170909/keypoint_train_images_20170902'

    tfrecord_dir = os.path.dirname(tfrecord_file)
    if not os.path.exists(tfrecord_dir):
        os.makedirs(tfrecord_dir)

    writer = tf.python_io.TFRecordWriter(tfrecord_file)
    total_img_nums = len(labels)
    count = 0
    count_zero = 0

    for label in labels:
        img_file = os.path.join(img_path, label['image_id'] + '.jpg')
        bbox = []
        category_id = []
        annotations = label['human_annotations']
        for key, value in annotations.items():
            #------convert box format [xmin, ymin, xmax, ymax] into [ymin, xmin, ymax, xmax]-------#
            if len(value) != 4:
                raise  ValueError('the box size must be equal to 4!!!!')
            tmp = value[0]
            value[0] = value[1]
            value[1] = tmp
            tmp = value[2]
            value[2] = value[3]
            value[3] = tmp
            #--------------------------------------------------------------------------------------#
            bbox += value
            category_id.append(1)
        if not bbox:
            count_zero += 1
            continue

        # ----if len(bboxs)//4 < n (set n = 20), add zeros to make len(bboxs)//4 == n------------#
        n = 30
        if len(bbox) < n * 4:
            last = n * 4 - len(bbox)
            bbox += list(np.zeros(last, dtype=np.float32))
            category_id += list(np.zeros(last // 4, dtype=np.int32))
        else:
            bbox = bbox[:n * 4]
            category_id = category_id[:n]
        # ----------------------------------------------------------------------------------------#

        img_data = tf.gfile.FastGFile(img_file, 'rb').read()
        img_dat = cv2.imread(img_file, cv2.IMREAD_COLOR)
        shape = img_dat.shape
        img_format = b'JPEG'

        # add to tfrecord
        example = tf.train.Example(features=tf.train.Features(
            feature={
                'image': _bytes_feature(img_data),
                'height': _int64_feature(shape[0]),
                'width': _int64_feature(shape[1]),
                'format': _bytes_feature(img_format),
                'channel': _int64_feature(shape[2]),
                'boxes': _float_feature(bbox),  # [xmin, ymin, xmax, ymax] * n
                'labels': _int64_feature(category_id)
            }
        ))
        writer.write(example.SerializeToString())
        count += 1

        # if count == 10:
        #     break
        if count % 1000 == 0:
            print ('Processing {}/{}'.format(count, total_img_nums))
    writer.close()
    print('Zeros box img nums {}'.format(count_zero))
    print('Convert tfrecord done.')

if __name__ == '__main__':

    convert_to_tfrecord(FLAGS.json_file, FLAGS.tfrecord_file)
    # convert_ai_challenger_tfrecord(FLAGS.tfrecord_file)