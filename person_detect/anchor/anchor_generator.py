from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from anchor import box_list_ops
from anchor import box_list

import tensorflow as tf
import numpy as np

class MultipleGridAnchorGenerator():
  """Generate a grid of anchors for multiple CNN layers."""

  def __init__(self,
               box_specs_list,
               base_anchor_sizes,
               clip_window=None):
    """Constructs a MultipleGridAnchorGenerator.

    To construct anchors, at multiple grid resolutions, one must provide a
    list of feature_map_shape_list (e.g., [(8, 8), (4, 4)]), and for each grid
    size, a corresponding list of (scale, aspect ratio) box specifications.

    For example:
    box_specs_list = [[(.1, 1.0), (.1, 2.0)],  # for 8x8 grid
                      [(.2, 1.0), (.3, 1.0), (.2, 2.0)]]  # for 4x4 grid

    To support the fully convolutional setting, we pass grid sizes in at
    generation time, while scale and aspect ratios are fixed at construction
    time.

    Args:
      box_specs_list: list of list of (scale, aspect ratio) pairs with the
        outside list having the same number of entries as feature_map_shape_list
        (which is passed in at generation time).
      base_anchor_sizes: list of base anchor size in each layer
      clip_window: a tensor of shape [4] specifying a window to which all
        anchors should be clipped. If clip_window is None, then no clipping
        is performed.

    Raises:
      ValueError: if box_specs_list is not a list of list of pairs
      ValueError: if clip_window is not either None or a tensor of shape [4]
    """
    if isinstance(box_specs_list, list) and all(
        [isinstance(list_item, list) for list_item in box_specs_list]):
      self._box_specs = box_specs_list
    else:
      raise ValueError('box_specs_list is expected to be a '
                       'list of lists of pairs')
    if isinstance(base_anchor_sizes, list):
        self._base_anchor_sizes = base_anchor_sizes
    else:
        raise ValueError('base_anchor_list is expected to be a list of float')
    if clip_window is not None and clip_window.get_shape().as_list() != [4]:
      raise ValueError('clip_window must either be None or a shape [4] tensor')
    self._clip_window = clip_window
    self._scales = []
    self._aspect_ratios = []
    for box_spec in self._box_specs:
      if not all([isinstance(entry, tuple) and len(entry) == 2
                  for entry in box_spec]):
        raise ValueError('box_specs_list is expected to be a '
                         'list of lists of pairs')
      scales, aspect_ratios = zip(*box_spec)
      self._scales.append(scales)
      self._aspect_ratios.append(aspect_ratios)

  def name_scope(self):
    return 'MultipleGridAnchorGenerator'

  def num_anchors_per_location(self):
    """Returns the number of anchors per spatial location.

    Returns:
      a list of integers, one for each expected feature map to be passed to
      the Generate function.
    """
    return [len(box_specs) for box_specs in self._box_specs]

  def generate(self,
                input_size,
                feature_map_shape_list,
                anchor_strides=None,
                anchor_offsets=None):
    """Generates a collection of bounding boxes to be used as anchors.

    The number of anchors generated for a single grid with shape MxM where we
    place k boxes over each grid center is k*M^2 and thus the total number of
    anchors is the sum over all grids. In our box_specs_list example
    (see the constructor docstring), we would place two boxes over each grid
    point on an 8x8 grid and three boxes over each grid point on a 4x4 grid and
    thus end up with 2*8^2 + 3*4^2 = 176 anchors in total. The layout of the
    output anchors follows the order of how the grid sizes and box_specs are
    specified (with box_spec index varying the fastest, followed by width
    index, then height index, then grid index).

    Args:
      input_size: input image size list with (width, height)
      feature_map_shape_list: list of pairs of conv net layer resolutions in the
        format [(height_0, width_0), (height_1, width_1), ...]. For example,
        setting feature_map_shape_list=[(8, 8), (7, 7)] asks for anchors that
        correspond to an 8x8 layer followed by a 7x7 layer.
      anchor_strides: list of pairs of strides (in y and x directions
        respectively). For example, setting
        anchor_strides=[(.25, .25), (.5, .5)] means that we want the anchors
        corresponding to the first layer to be strided by .25 and those in the
        second layer to be strided by .5 in both y and x directions. By
        default, if anchor_strides=None, then they are set to be the reciprocal
        of the corresponding grid sizes. The pairs can also be specified as
        dynamic tf.int or tf.float numbers, e.g. for variable shape input
        images.
      anchor_offsets: list of pairs of offsets (in y and x directions
        respectively). The offset specifies where we want the center of the
        (0, 0)-th anchor to lie for each layer. For example, setting
        anchor_offsets=[(.125, .125), (.25, .25)]) means that we want the
        (0, 0)-th anchor of the first layer to lie at (.125, .125) in image
        space and likewise that we want the (0, 0)-th anchor of the second
        layer to lie at (.25, .25) in image space. By default, if
        anchor_offsets=None, then they are set to be half of the corresponding
        anchor stride. The pairs can also be specified as dynamic tf.int or
        tf.float numbers, e.g. for variable shape input images.

    Returns:
      boxes: a BoxList holding a collection of N anchor boxes
    Raises:
      ValueError: if feature_map_shape_list, box_specs_list do not have the same
        length.
      ValueError: if feature_map_shape_list does not consist of pairs of
        integers
    """
    if not (isinstance(feature_map_shape_list, list)
            and len(feature_map_shape_list) == len(self._box_specs)):
      raise ValueError('feature_map_shape_list must be a list with the same '
                       'length as self._box_specs')
    if not all([isinstance(list_item, tuple) and len(list_item) == 2
                for list_item in feature_map_shape_list]):
      raise ValueError('feature_map_shape_list must be a list of pairs.')
    im_height, im_width = input_size[0], input_size[1]
    # anchor_strides = [(8.0, 8.0), (16.0, 16.0), (32.0, 32.0), (56.0, 56.0), (112.0, 112.0)]
    if not anchor_strides:
      anchor_strides = [(tf.to_float(im_height) / tf.to_float(pair[0]),
                         tf.to_float(im_width) / tf.to_float(pair[1]))
                        for pair in feature_map_shape_list]
    # anchor_offsets = [(4.0, 4.0), (8.0, 8.0), (16.0, 16.0), (28.0, 28.0), (56.0, 56.0)]
    if not anchor_offsets:
      anchor_offsets = [(0.5 * stride[0], 0.5 * stride[1])
                        for stride in anchor_strides]

    for arg, arg_name in zip([anchor_strides, anchor_offsets],
                             ['anchor_strides', 'anchor_offsets']):
      if not (isinstance(arg, list) and len(arg) == len(self._box_specs)):
        raise ValueError('%s must be a list with the same length '
                         'as self._box_specs' % arg_name)
      if not all([isinstance(list_item, tuple) and len(list_item) == 2
                  for list_item in arg]):
        raise ValueError('%s must be a list of pairs.' % arg_name)

    anchor_grid_list = []
    for grid_size, scales, aspect_ratios, stride, offset, base_anchor_size in zip(
        feature_map_shape_list, self._scales, self._aspect_ratios,
        anchor_strides, anchor_offsets, self._base_anchor_sizes):

      # print(grid_size, scales, aspect_ratios, stride, offset, base_anchor_size)

      anchor_grid_list.append(
          tile_anchors(
              grid_height=grid_size[0],
              grid_width=grid_size[1],
              scales=scales,
              aspect_ratios=aspect_ratios,
              base_anchor_size=base_anchor_size,
              anchor_stride=stride,
              anchor_offset=offset))
      # break
    concatenated_anchors = box_list_ops.concatenate(anchor_grid_list)
    num_anchors = concatenated_anchors.num_boxes_static()
    # print (num_anchors)
    if num_anchors is None:
      num_anchors = concatenated_anchors.num_boxes()
    if self._clip_window is not None:
      clip_window = tf.multiply(
          tf.to_float([im_height, im_width, im_height, im_width]),
          self._clip_window)
      concatenated_anchors = box_list_ops.clip_to_window(
          concatenated_anchors, clip_window, filter_nonoverlapping=False)
      # TODO: make reshape an option for the clip_to_window op
      concatenated_anchors.set(
          tf.reshape(concatenated_anchors.get(), [num_anchors, 4]))

    stddevs_tensor = 0.01 * tf.ones(
        [num_anchors, 4], dtype=tf.float32, name='stddevs')
    concatenated_anchors.add_field('stddev', stddevs_tensor)
    return concatenated_anchors


def tile_anchors(grid_height,
                 grid_width,
                 scales,
                 aspect_ratios,
                 base_anchor_size,
                 anchor_stride,
                 anchor_offset):
  """Create a tiled set of anchors strided along a grid in image space.

  This op creates a set of anchor boxes by placing a "basis" collection of
  boxes with user-specified scales and aspect ratios centered at evenly
  distributed points along a grid.  The basis collection is specified via the
  scale and aspect_ratios arguments.  For example, setting scales=[.1, .2, .2]
  and aspect ratios = [2,2,1/2] means that we create three boxes: one with scale
  .1, aspect ratio 2, one with scale .2, aspect ratio 2, and one with scale .2
  and aspect ratio 1/2.  Each box is multiplied by "base_anchor_size" before
  placing it over its respective center.

  Grid points are specified via grid_height, grid_width parameters as well as
  the anchor_stride and anchor_offset parameters.

  Args:
    grid_height: size of the grid in the y direction (int or int scalar tensor)
    grid_width: size of the grid in the x direction (int or int scalar tensor)
    scales: a 1-d  (float) tensor representing the scale of each box in the
      basis set.
    aspect_ratios: a 1-d (float) tensor representing the aspect ratio of each
      box in the basis set.  The length of the scales and aspect_ratios tensors
      must be equal.
    base_anchor_size: base anchor size in this layer as [height, width]
        (float tensor of shape [2])
    anchor_stride: difference in centers between base anchors for adjacent grid
                   positions (float tensor of shape [2])
    anchor_offset: center of the anchor with scale and aspect ratio 1 for the
                   upper left element of the grid, this should be zero for
                   feature networks with only VALID padding and even receptive
                   field size, but may need some additional calculation if other
                   padding is used (float tensor of shape [2])
  Returns:
    a BoxList holding a collection of N anchor boxes
  """
  ratio_sqrts = tf.sqrt(aspect_ratios)
  # 根据base_anchor_size计算anchor在原图上本来的宽高
  heights = scales / ratio_sqrts * base_anchor_size
  # print ('heights == ', heights.get_shape())
  widths = scales * ratio_sqrts * base_anchor_size
  # print ('widths == ', widths.get_shape())
  # Get a grid of box centers
  y_centers = tf.to_float(tf.range(grid_height))
  y_centers = y_centers * anchor_stride[0] + anchor_offset[0]
  # print ('y_centers before meshgrid === ', y_centers.get_shape())
  x_centers = tf.to_float(tf.range(grid_width))
  x_centers = x_centers * anchor_stride[1] + anchor_offset[1]
  # print('x_centers before meshgrid === ', x_centers.get_shape())
  x_centers, y_centers = tf.meshgrid(x_centers, y_centers)

  # xcenters在和widths进行meshgrid之前，xcenters的shape是(grid_height * grid_width)，只不过每一行都是0-（grid_width-1),widths长度为9，是总共要生成的
  # 9个anchors宽度列表，由前面计算得到。widths在和xcenters进行meshgrid之后，由于meshgrid是对维度为1的tensor进行操作，首先会把xcenters展开，
  # 变成一行，有(grid_height * grid_width)列，然后再进行meshgrid操作。meshgrid之后，widths_grid为 （grid_height * gird_width) × 9维矩阵，每一行都是9个anchor的宽度
  # xcenters_grid为(grid_height * grid_width) * 9矩阵，每一列都是grid_height个（0-grid_widht-1)数值。
  # 下面的heights和y_centers进行meshgrid最终得到的结果略有不同，heights_grid和widths_grid结果很一致，都是（grid_height * gird_width) × 9维矩阵，每一行都是9个anchor的高度，
  # 但y_centers_grid就略有变化，因为y_centers是每一列值都是 (0~grid_heigt-1),但每一行的值都是相同的，即每一行的值都是同一个值，meshgrid会将不是1维的矩阵变成一维，是按照行展开的，
  # 所以y_centers展开后就变成[1,1,1,1,1,..., 1,2,2,2,2,...,2,....,h,h,h,...h]这种形式，因此在和heights进行meshgrid之后，y_centers_grid每一列都变成了前面说的那个列表内的值

  widths_grid, x_centers_grid = tf.meshgrid(widths, x_centers)
  heights_grid, y_centers_grid = tf.meshgrid(heights, y_centers)

  # 在对y_centers_grid 和 x_centers_grid 进行axis=2的stack，x_centers_gird 和 y_centers_grid 维度均为  (grid_height*grid_width) * 9 维度，只不过数值不一样，按照
  # axis=2 进行stack，其实就是把x_centers_grid 和 y_centers_grid 里的值一一对应起来，最后变成 (grid_height * grid_width) * 9 * 2的三维矩阵，其实就是所有anchor对应的
  # 中心点在图像上的坐标，类似于[[[1,1]*9, [1,2]*9, ..., [7,7]*9]]这种形式，其实就是把图片上每个点的坐标拿出来，并重复9次，当做这个点生成的9个anchor的centers
  bbox_centers = tf.stack([y_centers_grid, x_centers_grid], axis=2)

  # 同理，对heights_grid 和 widths_grid 进行 axis=2 的stack， 也是得到一个（grid_height * grid_width) * 9 * 2的三维矩阵，只不过这个矩阵保存的是anchor的size，和前面的bbox_centers
  # 的值是一一对应的，即一个存了center的（x,y）坐标，一个存了bbox的宽高
  bbox_sizes = tf.stack([heights_grid, widths_grid], axis=2)

  # 接着对这两个矩阵进行reshape成 n*2 的二维矩阵，n是所有anchor的数量，为 (grid_heigt * grid_width * 9)，bbox_centers每一行保存的是anchor的中心点坐标
  # bbox_sizes 保存的是anchor的对应的宽高
  bbox_centers = tf.reshape(bbox_centers, [-1, 2])
  bbox_sizes = tf.reshape(bbox_sizes, [-1, 2])
  # convert [ycenter, xcenter, height, width] to [ymin, xmin, ymax, xmax]
  bbox_corners = _center_size_bbox_to_corners_bbox(bbox_centers, bbox_sizes)

  # 需要注意的是，这个生成的anchor就是相对于原图上的位置在哪，并且通过上一行的函数，把box的表示方式变成了[ymin, xmin, ymax, xmax]，最终的shape为(n, 4)
  # base_anchor_size 变得越来越大的原因是，随着featuremap维度不断增高，其上面的每一个点所能表示的原图的范围，也即是感受野也在不断增大
  return box_list.BoxList(bbox_corners)


def _center_size_bbox_to_corners_bbox(centers, sizes):
  """Converts bbox center-size representation to corners representation.

  Args:
    centers: a tensor with shape [N, 2] representing bounding box centers
    sizes: a tensor with shape [N, 2] representing bounding boxes

  Returns:
    corners: tensor with shape [N, 4] representing bounding boxes in corners
      representation
  """
  return tf.concat([centers - .5 * sizes, centers + .5 * sizes], 1)


def create_retinanet_anchors(
                       num_layers=5,
                       scales=(1.0, pow(2, 1./3), pow(2, 2./3)),
                       aspect_ratios=(0.5, 1.0, 2.0),
                       base_anchor_sizes=(32.0, 64.0, 128.0, 256.0, 512.0)
                       ):
    """Create a set of anchors walking along a grid in a collection of feature maps in RetinaNet.

    This op creates a set of anchor boxes by placing a basis collection of
    boxes with user-specified scales and aspect ratios centered at evenly
    distributed points along a grid. The basis  Each box is multiplied by
    base_anchor_size before placing it over its respective center.

    Args:
        num_layers: The number of grid layers to create anchors
        scales: A list of scales
        aspect_ratios: A list of aspect ratios
        base_anchor_sizes: List of base anchor sizes in each layer
    Returns:
        A MultipleGridAnchorGenerator
    """
    base_anchor_sizes = list(base_anchor_sizes)
    box_spec_list = []
    for idx in range(num_layers):
        layer_spec_list = []
        for scale in scales:
            for aspect_ratio in aspect_ratios:
                layer_spec_list.append((scale, aspect_ratio))
        box_spec_list.append(layer_spec_list)

    # for val in box_spec_list:
    #     print (val)
    # print (base_anchor_sizes)

    # box_spec_list = [[(1.0, 0.5), (1.0, 1.0), (1.0, 2.0),
    #                   (1.2599210498948732, 0.5), (1.2599210498948732, 1.0), (1.2599210498948732, 2.0),
    #                   (1.5874010519681994, 0.5), (1.5874010519681994, 1.0), (1.5874010519681994, 2.0)]]
    # base_anchor_sizes = [256.0]
    return MultipleGridAnchorGenerator(box_spec_list, base_anchor_sizes)


def anchor_assign(anchors, gt_boxes, gt_labels, is_training=True):
    """
    Assign generated anchors to boxes and labels
    Args:
        anchors: BoxList holding a collection of N anchors
        gt_boxes: BoxList holding a collection of groundtruth 2D box coordinates tensor/list [#object, 4]
            ([ymin, xmin, ymax, xmax], float type) of objects in given input image.
        gt_labels: Groundtruth 1D tensor/list [#object] (scalar int) of objects in given image.
        is_training: is training or not

    returns:
        BoxList with anchor location and class fields
    """
    pos_iou_thred = 0.5
    neg_iou_thred = 0.5
    if is_training:
        neg_iou_thred = 0.4
    if gt_boxes.get().get_shape()[0] != gt_labels.get_shape()[0]:
        raise ValueError('Boxs and labels number must be equal.')
    # box_iou: 总共有#anchors行，#gt_boxes列 (#anchors, #gtboxes)，每一行表示当前anchor对于gt_boxes的iou值
    box_iou = box_list_ops.iou(anchors, gt_boxes)

    # anchor_max_iou:  返回每一个anchor相对于gt_boxes中最大的iou值，
    # 是一个tensor，维度为[#anchors,], 每一个值为这个anchor和所有gtbox最大的iou值，
    # 和下面的anchor_max_iou_indices相对应
    anchor_max_iou = tf.reduce_max(box_iou, axis=1)

    # box_iou是一个二维矩阵，每一行代表一个anchor相对于gtbox的iou值，
    # 对其进行axis=1的tf.argmax,就是找到这个anchor和哪个gtbox iou值最大,并返回其下标
    anchor_max_iou_indices = tf.argmax(box_iou, axis=1)

    # 根据前面的anchor_max_iou_indices，将gt_boxes里对于每一个anchor是最大iou的那个gt_box取出来，
    # 组成一个新的矩阵，维度为[#anchors, 4]
    anchor_gt_box = tf.gather(gt_boxes.get(), anchor_max_iou_indices)

    # 类似于anchor_gt_box, 将前面anchor对应的最大iou的gt_box的label取出来，组成新的矩阵，维度为[#anchors,]
    anchor_gt_cls = tf.gather(gt_labels, anchor_max_iou_indices) #[#saved_anchor_num], 1D
    # print ('anchor_gt_cls === ', anchor_gt_cls)

    # get remaining index with iou between 0.4 to 0.5
    # 对于每一个anchor，因为其都有一个相对于gtbox的最大iou值，判断其是否是正样本，如果当前anchor的max_iou值大于pos_iou_thred,
    # 将其class设为其原本对应的label，否则设置为-1，为下一步操作做准备
    anchor_gt_cls = tf.where(tf.greater(anchor_max_iou, pos_iou_thred), anchor_gt_cls, 0-tf.ones_like(anchor_gt_cls))

    # 和上面的函数同理，如果anchor的max_iou小于neg_iou_thred,就将其class设置为0，否则就是原本的class
    # 因为已经考虑过其是否大于pos_iou_thred,所以执行完这个函数之后，最后得到的结果就是：
    # iou > 0.5的anchor认为是正样本，iou<0.4认为是负样本， iou在0.4和0.5之间设为-1，忽略掉
    anchor_gt_cls = tf.where(tf.less(anchor_max_iou, neg_iou_thred), tf.zeros_like(anchor_gt_cls), anchor_gt_cls)

    anchors.add_field('gt_boxes', anchor_gt_box)
    anchors.add_field('gt_labels', anchor_gt_cls) #[#saved_anchor_num], 1D
    return anchors

def anchor_test():
    input_size = [224,224]
    # feature_map = [(28, 28), (14, 14), (7, 7), (4, 4), (2, 2)]
    feature_maps = [(tf.ceil(input_size[0]/pow(2., i+3)), tf.ceil(input_size[1]/pow(2., i+3))) for i in range(5)]

    feature_map_list = [(tf.ceil(tf.multiply(tf.to_float(input_size[0]), 1 / pow(2., i + 3))),
                         tf.ceil(tf.multiply(tf.to_float(input_size[1]), 1 / pow(2., i + 3))))
                        for i in range(5)]

    # feature_map_list = [(3,3)]
    anchor_generator = create_retinanet_anchors()
    # print ('scales = ', anchor_generator._scales)
    # print ('aspect ratio = ', anchor_generator._aspect_ratios)

    anchors = anchor_generator.generate(input_size, feature_map_list)
    anchors_before_assign = anchors.get()
    # return
    gt_boxes = box_list.BoxList(tf.convert_to_tensor([[0, 0, 210, 210], [200,203,205,206], [1,1,220,220]], dtype=tf.float32))
    gt_labels = tf.convert_to_tensor([1, 1, 1])
    anchors, box_iou = anchor_assign(anchors, gt_boxes, gt_labels)
    # x = tf.convert_to_tensor([[[1,2,3],[3,4,5],[5,6,7]],[[1,2,3],[3,4,5],[5,6,7]]])
    result = anchors.get_field("gt_boxes")
    labels = anchors.get_field('gt_labels')
    print (labels.get_shape())
    with tf.Session() as sess:
        print (sess.run(result).shape)
        print (sess.run(labels).shape)
        print (sess.run(box_iou))
        # print(sess.run(tf.squeeze(tf.where(tf.greater(gt_labels, 1)))))
        # print(sess.run(tf.gather(x, tf.convert_to_tensor([0,1]), axis=1)))
    sess.close()

if __name__ == "__main__":
    anchor_test()