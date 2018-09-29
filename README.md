##This repository contains a TensorFlow implementation about this ECCV 2018 paper:

[Muhammed Kocabas, Salih Karagoz, Emre Akbas. MultiPoseNet: Fast Multi-Person Pose Estimation using Pose Residual Network. In ECCV, 2018.](https://arxiv.org/abs/1807.04067)

# This contains three part of this network:
    
- **keypoint_subnet**, use resnet_v2_50 + fpn as backbone net work, aiming to detect huaman pose points on a single image.

- **person_detect**, use as same as keypoint_subnet backbone, just a little different. Actually this part work is the RetinaNet, shown in paper [Focal Loss](https://arxiv.org/abs/1708.02002)

- **pose-residual-network**, the main contribution of this paper

Detailed information please see original [paper.](https://arxiv.org/abs/1807.04067)

**Note:** we trained three part network separately, just as in paper said, we first train keypoint_subnet and then frozen backbone parameters to trian person_detect sub_network. All training data is 
read through tf_record file.


###dataset: 

- pose-residual: ai_train2017.tfrecord ; coco_train2017.tfrecord
- person-detect: ai-instance-bbox.tfrecord ; coco-instance-bbox.tfrecord
- keypoint:
 
