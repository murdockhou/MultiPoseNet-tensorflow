# encoding: utf-8
'''
@author: shiwei hou
@contact: murdockhou@gmail.com
@software: PyCharm
@file: coco_json_convert.py
@time: 18-9-27 下午6:02

try to convert coco annotation json file into as like ai_challenger format
[
    {
        "image_id": "a0f6bdc065a602b7b84a67fb8d14ce403d902e0d",
        "human_annotations":
        {
            "human1": [178,250,290,522],
            "human2": [293,274,352,473],
            "human3": [315,236,389,495],
        ...},
        "keypoint_annotations":
        {
        "human1": [261, 294, 1, 281, 328, 1, 259, 314, 2,
                    213, 295, 1, 208, 346, 1, 192, 335, 1,
                    245, 375, 1, 255, 432, 1, 244, 494, 1,
                    221, 379, 1, 219, 442, 1, 226, 491, 1,
                    226, 256, 1, 231, 284, 1],
        "human2": [313, 301, 1, 305, 337, 1, 321, 345, 1,
                    331, 316, 2, 331, 335, 2, 344, 343, 2,
                    313, 359, 1, 320, 409, 1, 311, 454, 1,
                    327, 356, 2, 330, 409, 1, 324, 446, 1,
                    337, 284, 1, 327, 302, 1],
        "human3": [373, 304, 1, 346, 286, 1, 332, 263, 1,
                    363, 308, 2, 342, 327, 2, 345, 313, 1,
                    370, 385, 2, 368, 423, 2, 370, 466, 2,
                    363, 386, 1, 361, 424, 1, 361, 475, 1,
                    365, 273, 1, 369, 297, 1],
        ...}
    },
    ...
]
'''

import json
import numpy as np

coco_json_file = '/media/ulsee/E/datasets/coco/annotations2017/person_keypoints_train2017.json'

f      = open(coco_json_file, encoding='utf-8')
labels = json.load(f)
units  = []

img_info  = labels['images']
anno_info = labels['annotations']

print ('Start converting json file.....')
ll    = len(img_info)
count = 0

for img in img_info:
    unit     = {}
    img_name = img['file_name'].split('.')[0]
    img_id   = img['id']
    height   = img['height']
    width    = img['width']

    keypoint_anno = {}
    human_anno    = {}
    human_count   = 0

    for anno in anno_info:
        bbox        = anno['bbox']
        anno_img_id = anno['image_id']
        keypoints   = anno['keypoints']
        category_id = anno['category_id']

        if anno_img_id == img_id:
            bbox[2] = bbox[0] + bbox[2]
            bbox[3] = bbox[1] + bbox[3]
            keypoint_anno['human'+str(human_count)] = keypoints
            human_anno['human'+str(human_count)]    = bbox
            human_count += 1
    if human_count == 0:
        continue
    unit['image_id']             = img_name
    unit['keypoint_annotations'] = keypoint_anno
    unit['human_annotations']    = human_anno

    units.append(unit)

    count += 1

    # if count == 10:
    #     break

    if count % 100 == 0:
        print ('Processing {}/{}'.format(count, ll))

with open('/media/ulsee/E/pose_residual_net_tfrecord/cocotrain2017_convert_ai.json', 'w') as fw:
    json.dump(units, fw)
    print ('Convert done.')





