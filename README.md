### NOTE:

I find somewhere is weird in eval.py in the official [PRN-pytorch impementaion repo](https://github.com/salihkaragoz/pose-residual-network-pytorch). When to get predicated bbox_keypoints, the code used the true keypoints to assign the bbox_keypoints. The code in eval.py is about line 200 and line 205. The peaks is true keypoints coordinate, it seems that used the true coordinate to assign the predicated bbox_keypoints. Actually i think the line 209~220 in eval.py is the right way to get real predicated bbox_keypoints.

As far as i can see, i think that this ropo has some problems and cann't get the correct result through 'correct way'. But the author did not response to me and maybe there are still some tricks in this repo that i didn't found yet.

## This repository contains a TensorFlow implementation about this ECCV 2018 paper:

[Muhammed Kocabas, Salih Karagoz, Emre Akbas. MultiPoseNet: Fast Multi-Person Pose Estimation using Pose Residual Network. In ECCV, 2018.](https://arxiv.org/abs/1807.04067)

# This contains three part of this network:
    
- **keypoint_subnet**, use resnet_v2_50 + fpn as backbone net work, aiming to detect huaman pose points on a single image.

- **person_detect**, use as same as keypoint_subnet backbone, just a little different. Actually this part work is the RetinaNet, shown in paper [Focal Loss](https://arxiv.org/abs/1708.02002)

- **pose-residual-network**, the main contribution of this paper

Detailed information please see original [paper.](https://arxiv.org/abs/1807.04067)

**Note:** we trained three part network separately, just as in paper said, we first train keypoint_subnet and then frozen backbone parameters to trian person_detect sub_network. All training data is 
read through tf_record file.


### dataset: 

- pose-residual: ai_train2017.tfrecord ; coco_train2017.tfrecord
- person-detect: ai-instance-bbox.tfrecord ; coco-instance-bbox.tfrecord
- keypoint: ai_train2017.tfrecord & ai_train2017.json ; coco_train2017.tfrecord & coco_train2017.json

coco-keypoints-annotations:

[0-16]::::::[nose, left_eye, right_eye, left_ear, right_ear, left_shoulder, right_shoulder, left_elbow,
 right_elbow, left_wrist, right_wrist, left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle]

 # Thanks
 
 [mkocabas](https://github.com/mkocabas/pose-residual-network)
 [salihkaragoz](https://github.com/salihkaragoz/pose-residual-network-pytorch)
 
 
