next: 考虑使用softargmax对heatmap进行求和得到关键点，单人的比较直观，每个heatmap直接生成一个关节点即可。多人的需要考虑下，是事先固定好关节点个数，或者其它方式

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
 
 
