#PRN 网络理解：

PRN网络的思想，就是对一个bounding box里，如果有多个相同部位的关键点出现在一个single box里，也就是有多个人，那么就很难判定这么多个关键点到底属于哪一个人。PRN的思想就是，一个single box就是一个人，一个人应该只有一种类型的关键点，前面keypoint Subnet网络得到的关键点位置，如果在这个single box范围内，那么就将这个single box范围内的关键点作为输入放进去PRN网络，经过计算之后，PRN网络对每个channel只输出一个关键点，并且认为这个关键点就是这个single box框起来的人的关键点。

#PRN网络训练的数据构造

PRN网络是对单独的一个一个box进行训练的，而不是一张图片。


- label： PRN网络的label就是一个对gt_box处理过后的ROI。论文里将box缩放为高宽为56*36，height/width = 1.56的ROI。将此作为网络的输入、输出大小。如果有17个关键点要训练，那么label大小就是[56, 36, 17],每个channel的意义和使用keypoint subnet得到的输出意义一致。首先就是对gt_box里所有关键点进行处理，和keypoint subnet进行训练时一样，对每个关键点出现的位置，在label对应的位置上打上标签1，否则就是0，其实就是一个heatmap，只不过是以box生成的heatmap。
- input： PRN网络的input就是预先设定好的box大小，首先将input全部设为0，然后对于这个box所在的图片上，所有出现过的关键点进行处理。和生成label过程一样，只不过处理的关键点不仅仅是原本属于gt_box的关键点了，而是这个图片上所有的出现在gt_box范围内的关键点，同样使用生成label的方法，生成网络的input。


label和input生成之后，均进行高斯处理(sigma小于1显示比较明显)，最后得到的结果才是PRN网络的输入和label。

### 训练结果：

- 和官方提供的pytorch版本一致，训练参数一致，训练次数一致，在coco val2017的结果如下：

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.886
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.977
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.920
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.874
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.912
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.911
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.980
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.933
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.893
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.943

```
 使用官方提供的pytorch版本训练，使用提供的参数和数据，在coco val2017的结果如下：
 
 ```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.888
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.977
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.920
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.876
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.910
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.913
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.981
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.933
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.894
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.943

```
   官方宣称能达到的精度如下：
 ```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.892
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.978
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.921
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.883
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.912
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.917
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.982
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.937
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.902
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.944

```