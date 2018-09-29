# encoding: utf-8
'''
@author: shiwei hou
@contact: murdockhou@gmail.com
@software: PyCharm
@file: json_read.py
@time: 18-8-28 下午3:11
'''
import json

def load_json(json_file):
    '''
    load json file and return a dict, like ['id'] = [keypoints], typically used for ai_challenger format dataset
    :param json_file:
    :return:
    '''
    f = open(json_file, encoding='utf-8')
    labels = json.load(f)
    label_dict = {}
    for label in labels:
        current_keypoints = []
        for human, keypoints in label['keypoint_annotations'].items():
            current_keypoints = current_keypoints + keypoints
        label_dict[label['image_id']] = current_keypoints
    return label_dict

def load_coco_json(json_file):
    '''

    :param json_file:
    :return:
    '''
    f = open(json_file, encoding='utf-8')
    labels = json.load(f)
    return labels

def dump_coco_data(json_file):
    '''
    convert coco annotatinos json file, as like:[{'image_id":keypoints}]
    :param json_file:
    :return:
    '''

    f = open(json_file, encoding='utf-8')
    labels = json.load(f)
    image_info = labels['images']
    anno_info = labels['annotations']
    label_dict = {}

    for image in image_info:
        image_name = image['file_name'].split('.')[0]
        image_id = image['id']
        current_keypoints = []
        for anno in anno_info:
            keypoints = anno['keypoints']
            anno_image_id = anno['image_id']
            anno_id = anno['id']
            if anno_image_id == image_id:
                current_keypoints = current_keypoints + keypoints

        label_dict[image_name] = current_keypoints
    with open('coco_image_name_to_keypoints.json', 'w') as fw:
        json.dump(label_dict, fw)

def convert_coco_instance_json(json_file):
    '''
    convert coco annotatinos json file, as like:[{'image_id":[x1, y1, w, h, category_id] * n}]
    :param json_file:
    :return:
    '''

    f = open(json_file, encoding='utf-8')
    labels = json.load(f)
    units = {}

    image_info = labels['images']
    anno_info = labels['annotations']
    print ('start reading json......')
    ll = len(image_info)
    count  = 1
    for image in image_info:
        image_name = image['file_name'].split('.')[0]
        image_id = image['id']
        height = image['height']
        width = image['width']
        current_bbox = [height, width]

        for anno in anno_info:
            bbox = anno['bbox']
            anno_image_id = anno['image_id']

            if anno_image_id == image_id:
                bbox.append(anno['category_id'])
                current_bbox = current_bbox + bbox
        units[image_name] = current_bbox

        if count % 1000 == 0:
            print ('Processing {}'.format(count/ll))
        count += 1
        if count == 10:
            break

    is_save = True
    if is_save:
        save_json_file = 'coco-instance-imgid-bbox.json'

        with open(save_json_file, 'w') as fw:
            json.dump(units, fw)
if __name__ == '__main__':
    convert_coco_instance_json('/media/ulsee/E/datasets/coco-annotations/instances_train2017.json')