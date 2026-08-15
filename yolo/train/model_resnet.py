import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.layers import UpSampling2D,Concatenate,LeakyReLU
from tensorflow.keras.models import Model
import math

backbone_layer_name = []
save_layer = True

def Relu6(x,max_value=6):
    return tf.keras.activations.relu(x,max_value=6)


def _conv_block(inputs, filters, kernel, strides):
    """Convolution Block
    This function defines a 2D convolution operation with BN and relu6.

    # Arguments
        inputs: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution along the width and height.
            Can be a single integer to specify the same value for
            all spatial dimensions.

    # Returns
        Output tensor.
    """
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    x = tf.keras.layers.Conv2D(filters, kernel, padding='same', strides=strides)(inputs)
    x = tf.keras.layers.BatchNormalization(axis=channel_axis)(x)
    return Relu6(x)


def _bottleneck(inputs, filters, kernel, t, s, r=False):
    """Bottleneck
    This function defines a basic bottleneck structure.

    # Arguments
        inputs: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        t: Integer, expansion factor.
            t is always applied to the input size.
        s: An integer or tuple/list of 2 integers,specifying the strides
            of the convolution along the width and height.Can be a single
            integer to specify the same value for all spatial dimensions.
        r: Boolean, Whether to use the residuals.

    # Returns
        Output tensor.
    """
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    tchannel = inputs.shape[channel_axis] * t

    x = _conv_block(inputs, tchannel, (1, 1), (1, 1))

    x = tf.keras.layers.DepthwiseConv2D(kernel, strides=(s, s), depth_multiplier=1, padding='same')(x)
    x = tf.keras.layers.BatchNormalization(axis=channel_axis)(x)
    x = Relu6(x)

    x = tf.keras.layers.Conv2D(filters, (1, 1), strides=(1, 1), padding='same')(x)
    x = tf.keras.layers.BatchNormalization(axis=channel_axis)(x)

    if r:
        x = tf.keras.layers.add([x, inputs])
    return x


def _inverted_residual_block(inputs, filters, kernel, t, strides, n):
    """Inverted Residual Block
    This function defines a sequence of 1 or more identical layers.

    # Arguments
        inputs: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        t: Integer, expansion factor.
            t is always applied to the input size.
        s: An integer or tuple/list of 2 integers,specifying the strides
            of the convolution along the width and height.Can be a single
            integer to specify the same value for all spatial dimensions.
        n: Integer, layer repeat times.
    # Returns
        Output tensor.
    """

    x = _bottleneck(inputs, filters, kernel, t, strides)

    for i in range(1, n):
        x = _bottleneck(x, filters, kernel, t, 1, True)

    return x

def Conv(x, filter, filter_shape = (3,3), stride=2, has_relu = True,has_leaky_relu = False):
    _x = tf.keras.layers.Conv2D(filter, filter_shape, strides=stride, padding="same")(x)
    if save_layer: backbone_layer_name.append(_x.name)
    _x = tf.keras.layers.BatchNormalization()(_x)
    if has_leaky_relu: _x = LeakyReLU(alpha=0.1)(_x)
    elif has_relu: _x = Relu6(_x)
    return _x

def dsConv(x, filter_shape = (3, 3), stride=2):
    _x = tf.keras.layers.DepthwiseConv2D(filter_shape, strides=stride, padding="same")(x)
    if save_layer: backbone_layer_name.append(_x.name)
    _x = tf.keras.layers.BatchNormalization()(_x)
    _x = Relu6(_x)
    return _x


def mbv2_body(input):
    stride = 1
    x1 = _conv_block(input, 32, (5, 5), strides=(2, 2))
    stride = stride * 2
    x1 = _inverted_residual_block(x1, 16, (3, 3), t=1, strides=2, n=1)
    stride = stride * 2
    x1 = _inverted_residual_block(x1, 24, (3, 3), t=2, strides=1, n=2)
    x1 = _inverted_residual_block(x1, 32, (3, 3), t=3, strides=2, n=2)
    stride = stride * 2
    #x1 = _inverted_residual_block(x1, 64, (3, 3), t=6, strides=4, n=2)
    #x1 = _inverted_residual_block(x1, 96, (3, 3), t=6, strides=3, n=1)
    #x1 = _inverted_residual_block(x1, 160, (3, 3), t=6, strides=3, n=2)
    #x1 = _inverted_residual_block(x1, 320, (3, 3), t=6, strides=1, n=1)
    #x1 = Conv(x1, 1280, (1,1), 1)

    return x1,stride

def tiny_yolo_res_body(shape, num_anchors, num_classes,num_heads,divider):
    input = tf.keras.layers.Input(shape=shape)
    x1,stride = mbv2_body(input)

    if(num_heads == 2):
        #//small header
        if(divider[1] > stride):
            next_stride = 2
        else:
            next_stride = 1
        stride = stride * next_stride
        x1 = _bottleneck(x1,128,(5,5),1,next_stride,False)

        #//big header
        if(divider[0] > stride):
            next_stride = 2
        else:
            next_stride = 1
        stride = stride* next_stride
        x2 = _inverted_residual_block(x1,64,(3,3),t=2,strides=next_stride,n=2)
        '''
        x2_ = Conv(x2, 128, (5,5), 1)
        x2_ = Conv(x2, 128, (1,1), 1)
        '''
        x2_ = _bottleneck(x2,128,(5,5),1,1,False)
        y1 = tf.keras.layers.Conv2D(num_anchors*(num_classes+5), (1,1), strides=1, padding="same")(x2_)
        
        x2 = Conv(x2, 128, (1,1), 1)
        x2 = UpSampling2D(2)(x2)
        y2 = Concatenate()([x2, x1])    
        
        '''
        y2 = Conv(y2, 128, (5,5), 1)
        y2 = Conv(y2, 128, (3,3), 1)
        y2 = Conv(y2, 128, (1,1), 1)
        '''
        y2 = _bottleneck(y2,128,(5,5),1,1,False)
        y2 = tf.keras.layers.Conv2D(num_anchors*(num_classes+5), (1,1), strides=1, padding="same")(y2)

        return Model(input, [y1,y2]) 
    else:
        if(divider[0] == 16):
            stride = 2
        else:
            stride = 1
        '''
        y1 = Conv(x1, 128, (3,3), stride)
        y1 = Conv(y1, 128, (3,3), 1)
        #y1 = Conv(y1, 128, (3,3), 1)
        y1 = Conv(y1, 128, (1,1), 1)
        '''
        x1 = _inverted_residual_block(x1,32,(5,5),t=2,strides=stride,n=2)
        y1 = _bottleneck(x1,128,(5,5),2,1,False)
        y1 = tf.keras.layers.Conv2D(num_anchors*(num_classes+5), (1,1), strides=1, padding="same")(y1)

    return Model(input, y1)


def yolo_head(feats, anchors, num_classes, input_shape, calc_loss=False):
    """Convert final layer features to bounding box parameters."""
    num_anchors = len(anchors)
    # Reshape to batch, height, width, num_anchors, box_params.
    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2])
    
    grid_shape = K.shape(feats)[1:3] # height, width
    grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
        [1, grid_shape[1], 1, 1])
    grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
        [grid_shape[0], 1, 1, 1])
    grid = K.concatenate([grid_x, grid_y])
    try:
        grid = K.cast(grid, K.dtype(feats))
    except:
        grid = K.cast(grid, K.dtype(K.constant(feats)))

    feats = K.reshape(
        feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

    # Adjust preditions to each spatial grid point and anchor size.
    box_xy = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[::-1], K.dtype(feats))
    box_wh = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[::-1], K.dtype(feats))
    box_confidence = K.sigmoid(feats[..., 4:5])
    box_class_probs = K.sigmoid(feats[..., 5:])

    if calc_loss == True:
        return grid, feats, box_xy, box_wh
    return box_xy, box_wh, box_confidence, box_class_probs


def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape):
    '''Get corrected boxes'''
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = K.cast(input_shape, K.dtype(box_yx))
    image_shape = K.cast(image_shape, K.dtype(box_yx))
    new_shape = K.round(image_shape * K.min(input_shape/image_shape))
    offset = (input_shape-new_shape)/2./input_shape
    scale = input_shape/new_shape
    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes =  K.concatenate([
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ])

    # Scale boxes back to original image shape.
    boxes *= K.concatenate([image_shape, image_shape])
    return boxes


def yolo_boxes_and_scores(feats, anchors, num_classes, input_shape, image_shape):
    '''Process Conv layer output'''
    box_xy, box_wh, box_confidence, box_class_probs = yolo_head(feats,
        anchors, num_classes, input_shape)
    boxes = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape)
    boxes = K.reshape(boxes, [-1, 4])
    box_scores = box_confidence * box_class_probs
    box_scores = K.reshape(box_scores, [-1, num_classes])
    return boxes, box_scores


def yolo_eval(yolo_outputs,
              anchors,
              num_classes,
              image_shape,
              input_shape,
              max_boxes=20,
              score_threshold=.6,
              iou_threshold=.5):
    """Evaluate YOLO model on given input and return filtered boxes."""
    num_layers = len(yolo_outputs)

    if (num_layers>1):
        anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5],[0,1,2]] # default setting
        boxes = []
        box_scores = []

        for l in range(num_layers):
            _boxes, _box_scores = yolo_boxes_and_scores(yolo_outputs[l],
                anchors[anchor_mask[l]], num_classes, input_shape, image_shape)
            boxes.append(_boxes)
            box_scores.append(_box_scores)
    else:
        anchor_mask = [[0,1,2]]
        input_shape = K.shape(yolo_outputs[0])[1:3] * 8
        boxes, box_scores = yolo_boxes_and_scores(yolo_outputs[0],
            anchors[anchor_mask[0]], num_classes, input_shape, image_shape)
        boxes = [boxes]
        box_scores = [box_scores]
    
    boxes = K.concatenate(boxes, axis=0)
    box_scores = K.concatenate(box_scores, axis=0)

    mask = box_scores >= score_threshold
    max_boxes_tensor = K.constant(max_boxes, dtype='int32')
    boxes_ = []
    scores_ = []
    classes_ = []
    for c in range(num_classes):
        # TODO: use keras backend instead of tf.
        class_boxes = tf.boolean_mask(boxes, mask[:, c])
        class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
        np_class_box_scores = class_box_scores.numpy()
        nms_index = tf.image.non_max_suppression(
            class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=iou_threshold)
        class_boxes = K.gather(class_boxes, nms_index)
        class_box_scores = K.gather(class_box_scores, nms_index)
        classes = K.ones_like(class_box_scores, 'int32') * c
        boxes_.append(class_boxes)
        scores_.append(class_box_scores)
        classes_.append(classes)
    boxes_ = K.concatenate(boxes_, axis=0)
    scores_ = K.concatenate(scores_, axis=0)
    classes_ = K.concatenate(classes_, axis=0)

    return boxes_, scores_, classes_


def box_iou(b1, b2):
    '''Return iou tensor

    Parameters
    ----------
    b1: tensor, shape=(i1,...,iN, 4), xywh
    b2: tensor, shape=(j, 4), xywh

    Returns
    -------
    iou: tensor, shape=(i1,...,iN, j)

    '''

    # Expand dim to apply broadcasting.
    b1 = K.expand_dims(b1, -2)
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh/2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half

    # Expand dim to apply broadcasting.
    b2 = K.expand_dims(b2, 0)
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh/2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    intersect_mins = K.maximum(b1_mins, b2_mins)
    intersect_maxes = K.minimum(b1_maxes, b2_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    iou = intersect_area / (b1_area + b2_area - intersect_area)

    return iou

def box_iou_loss(b1, b2,iou_type='siou'):
    '''Return iou tensor

    Parameters
    ----------
    b1: tensor, shape=(i1,...,iN, 4), xywh
    b2: tensor, shape=(j, 4), xywh

    Returns
    -------
    iou: tensor, shape=(i1,...,iN, j)

    '''


    b1_xy       = b1[..., :2]
    b1_wh       = b1[..., 2:4]
    b1_wh_half  = b1_wh/2.
    b1_mins     = b1_xy - b1_wh_half
    b1_maxes    = b1_xy + b1_wh_half

    b2_xy       = b2[..., :2]
    b2_wh       = b2[..., 2:4]
    b2_wh_half  = b2_wh/2.
    b2_mins     = b2_xy - b2_wh_half
    b2_maxes    = b2_xy + b2_wh_half
    #-----------------------------------------------------------#
    #   求真实框和预测框所有的iou
    #   iou         (batch, feat_w, feat_h, anchor_num)
    #-----------------------------------------------------------#
    intersect_mins  = K.maximum(b1_mins, b2_mins)
    intersect_maxes = K.minimum(b1_maxes, b2_maxes)
    intersect_wh    = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area  = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area         = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area         = b2_wh[..., 0] * b2_wh[..., 1]
    union_area      = b1_area + b2_area - intersect_area
    iou             = intersect_area / K.maximum(union_area, K.epsilon())

    center_wh       = b1_xy - b2_xy
    #----------------------------------------------------#
    #   找到包裹两个框的最小框的左上角和右下角
    #----------------------------------------------------#
    enclose_mins    = K.minimum(b1_mins, b2_mins)
    enclose_maxes   = K.maximum(b1_maxes, b2_maxes)
    enclose_wh      = K.maximum(enclose_maxes - enclose_mins, 0.0)

    if iou_type == 'ciou':
        #-----------------------------------------------------------#
        #   计算中心的差距
        #   center_distance (batch, feat_w, feat_h, anchor_num)
        #-----------------------------------------------------------#
        center_distance = K.sum(K.square(center_wh), axis=-1)
        #-----------------------------------------------------------#
        #   计算对角线距离
        #   enclose_diagonal (batch, feat_w, feat_h, anchor_num)
        #-----------------------------------------------------------#
        enclose_diagonal = K.sum(K.square(enclose_wh), axis=-1)
        ciou    = iou - 1.0 * (center_distance) / K.maximum(enclose_diagonal, K.epsilon())
        
        v       = 4 * K.square(tf.math.atan2(b1_wh[..., 0], K.maximum(b1_wh[..., 1], K.epsilon())) - tf.math.atan2(b2_wh[..., 0], K.maximum(b2_wh[..., 1],K.epsilon()))) / (math.pi * math.pi)
        alpha   = v /  K.maximum((1.0 - iou + v), K.epsilon())
        out     = ciou - alpha * v
    elif iou_type == 'siou':
        #----------------------------------------------------#
        #   Angle cost
        #----------------------------------------------------#
        #----------------------------------------------------#
        #   计算中心的距离
        #----------------------------------------------------#
        sigma       = tf.pow(center_wh[..., 0] ** 2 + center_wh[..., 1] ** 2, 0.5)
        
        #----------------------------------------------------#
        #   求h和w方向上的sin比值
        #----------------------------------------------------#
        sin_alpha_1 = tf.abs(center_wh[..., 0]) / K.maximum(sigma, K.epsilon())
        sin_alpha_2 = tf.abs(center_wh[..., 1]) / K.maximum(sigma, K.epsilon())

        #----------------------------------------------------#
        #   求门限，二分之根号二，0.707
        #   如果门限大于0.707，代表某个方向的角度大于45°
        #   此时取另一个方向的角度
        #----------------------------------------------------#
        threshold   = pow(2, 0.5) / 2
        sin_alpha   = tf.where(sin_alpha_1 > threshold, sin_alpha_2, sin_alpha_1)

        #----------------------------------------------------#
        #   alpha越接近于45°，angle_cost越接近于1，gamma越接近于1
        #   alpha越接近于0°，angle_cost越接近于0，gamma越接近于2
        #----------------------------------------------------#
        angle_cost  = tf.cos(tf.asin(sin_alpha) * 2 - math.pi / 2)
        gamma       = 2 - angle_cost
        
        #----------------------------------------------------#
        #   Distance cost
        #   求中心与外包围举行高宽的比值
        #----------------------------------------------------#
        rho_x           = (center_wh[..., 0] / K.maximum(enclose_wh[..., 0], K.epsilon())) ** 2
        rho_y           = (center_wh[..., 1] / K.maximum(enclose_wh[..., 1], K.epsilon())) ** 2
        distance_cost   = 2 - tf.exp(-gamma * rho_x) - tf.exp(-gamma * rho_y)
        
        #----------------------------------------------------#
        #   Shape cost
        #   真实框与预测框的宽高差异与最大值的比值
        #   差异越小，costshape_cost越小
        #----------------------------------------------------#
        omiga_w     = tf.abs(b1_wh[..., 0] - b2_wh[..., 0]) / K.maximum(tf.maximum(b1_wh[..., 0], b2_wh[..., 0]), K.epsilon())
        omiga_h     = tf.abs(b1_wh[..., 1] - b2_wh[..., 1]) / K.maximum(tf.maximum(b1_wh[..., 1], b2_wh[..., 1]), K.epsilon())
        shape_cost  = tf.pow(1 - tf.exp(-1 * omiga_w), 4) + tf.pow(1 - tf.exp(-1 * omiga_h), 4)
        out         = iou - 0.5 * (distance_cost + shape_cost)

    return K.expand_dims(out, -1)

def yolo_loss_iou(args, anchors, num_classes, ignore_thresh=.5, print_loss=False, obj_scale=1,noobj_scale=1, iou_type ='siou'):
    '''Return yolo_loss tensor

    Parameters
    ----------
    yolo_outputs: list of tensor, the output of yolo_body or tiny_yolo_body
    y_true: list of array, the output of preprocess_true_boxes
    anchors: array, shape=(N, 2), wh
    num_classes: integer
    ignore_thresh: float, the iou threshold whether to ignore object confidence loss

    Returns
    -------
    loss: tensor, shape=(1,)

    '''
    balance         = [0.4, 1.0, 4], 
    box_ratio       = 0.05, 
    obj_ratio       = 1
    cls_ratio       = 0.5 / 4,

    num_layers = len(anchors)//3 # default setting
    yolo_outputs = args[:num_layers]
    y_true = args[num_layers:]
    if(num_layers > 1):
        anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5],[0,1,2]]
        input_shape = K.cast(K.shape(yolo_outputs[0])[1:3] * 32, K.dtype(y_true[0]))
        grid_shapes = [K.cast(K.shape(yolo_outputs[l])[1:3], K.dtype(y_true[0])) for l in range(num_layers)]
        m = K.shape(yolo_outputs[0])[0] # batch size, tensor
        mf = K.cast(m, K.dtype(yolo_outputs[0]))
    else:
        anchor_mask = [[0,1,2]]
        input_shape = K.cast(K.shape(yolo_outputs[0])[1:3] * 32, K.dtype(y_true[0]))
        grid_shapes = K.cast(K.shape(yolo_outputs[0])[1:3], K.dtype(y_true[0]))
        m = K.shape(yolo_outputs[0])[0] # batch size, tensor
        mf = K.cast(m, K.dtype(yolo_outputs[0]))

    
    loss = 0
    
    
    for l in range(num_layers):
        #tf.print("\r\r****** l anchors:,grid_shapes",l,anchors[anchor_mask[l][0]])
        object_mask = y_true[l][..., 4:5]
        true_class_probs = y_true[l][..., 5:]

        grid, raw_pred, pred_xy, pred_wh = yolo_head(yolo_outputs[l],
             anchors[anchor_mask[l]], num_classes, input_shape, calc_loss=True)
        pred_box = K.concatenate([pred_xy, pred_wh])

        # Find ignore mask, iterate over each of batch.
        ignore_mask = tf.TensorArray(K.dtype(y_true[0]), size=1, dynamic_size=True)
        object_mask_bool = K.cast(object_mask, 'bool')
        def loop_body(b, ignore_mask):
            true_box = tf.boolean_mask(y_true[l][b,...,0:4], object_mask_bool[b,...,0])
            iou = box_iou(pred_box[b], true_box)
            best_iou = K.max(iou, axis=-1)
            try:
                tf.print("@@@train:",np.max(best_iou.numpy()),np.average(best_iou.numpy()))
            except:
                print()
            ignore_mask = ignore_mask.write(b, K.cast(best_iou<ignore_thresh, K.dtype(true_box)))
            return b+1, ignore_mask
        
        _, ignore_mask = tf.while_loop(lambda b,*args: b<m, loop_body, [0, ignore_mask])
        ignore_mask = ignore_mask.stack()
        ignore_mask = K.expand_dims(ignore_mask, -1)

        # K.binary_crossentropy is helpful to avoid exp overflow.
        
        raw_true_box    = y_true[l][...,0:4]
        iou             = box_iou_loss(pred_box, raw_true_box, iou_type)
        iou_loss        = object_mask * (1 - iou)
        location_loss   = K.sum(iou_loss)

        confidence_loss = obj_scale * object_mask * K.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True)+ \
            noobj_scale * (1-object_mask) * K.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True) * ignore_mask
        class_loss = object_mask * K.binary_crossentropy(true_class_probs, raw_pred[...,5:], from_logits=True)

        #-----------------------------------------------------------#
        #   计算正样本数量
        #-----------------------------------------------------------#
        num_pos         = tf.maximum(K.sum(K.cast(object_mask, tf.float32)), 1)
        num_neg         = tf.maximum(K.sum(K.cast((1 - object_mask) * ignore_mask, tf.float32)), 1)

        #-----------------------------------------------------------#
        #   将所有损失求和
        #-----------------------------------------------------------#
        location_loss   = location_loss * box_ratio / num_pos
        confidence_loss = K.sum(confidence_loss) * balance[l] * obj_ratio / (num_pos + num_neg)
        class_loss      = K.sum(class_loss) * cls_ratio / num_pos / num_classes

        loss            += location_loss + confidence_loss + class_loss
        if print_loss:
            tf.print("\n-----loss:",l,location_loss, confidence_loss, class_loss, K.max(K.sigmoid(raw_pred[...,4:5])))
            #tf.print('\n---\n')
    return loss

def yolo_loss(args, anchors, num_classes, ignore_thresh=.5, print_loss=False, obj_scale=1,noobj_scale=1):
    '''Return yolo_loss tensor

    Parameters
    ----------
    yolo_outputs: list of tensor, the output of yolo_body or tiny_yolo_body
    y_true: list of array, the output of preprocess_true_boxes
    anchors: array, shape=(N, 2), wh
    num_classes: integer
    ignore_thresh: float, the iou threshold whether to ignore object confidence loss

    Returns
    -------
    loss: tensor, shape=(1,)

    '''
    num_layers = len(anchors)//3 # default setting
    yolo_outputs = args[:num_layers]
    y_true = args[num_layers:]
    if(num_layers > 1):
        anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5],[0,1,2]]
        input_shape = K.cast(K.shape(yolo_outputs[0])[1:3] * 32, K.dtype(y_true[0]))
        grid_shapes = [K.cast(K.shape(yolo_outputs[l])[1:3], K.dtype(y_true[0])) for l in range(num_layers)]
        m = K.shape(yolo_outputs[0])[0] # batch size, tensor
        mf = K.cast(m, K.dtype(yolo_outputs[0]))
    else:
        anchor_mask = [[0,1,2]]
        input_shape = K.cast(K.shape(yolo_outputs[0])[1:3] * 32, K.dtype(y_true[0]))
        grid_shapes = K.cast(K.shape(yolo_outputs[0])[1:3], K.dtype(y_true[0]))
        m = K.shape(yolo_outputs[0])[0] # batch size, tensor
        mf = K.cast(m, K.dtype(yolo_outputs[0]))

    
    loss = 0
    
    
    for l in range(num_layers):
        #tf.print("\r\r****** l anchors:,grid_shapes",l,anchors[anchor_mask[l][0]])
        object_mask = y_true[l][..., 4:5]
        true_class_probs = y_true[l][..., 5:]

        grid, raw_pred, pred_xy, pred_wh = yolo_head(yolo_outputs[l],
             anchors[anchor_mask[l]], num_classes, input_shape, calc_loss=True)
        pred_box = K.concatenate([pred_xy, pred_wh])

        # Darknet raw box to calculate loss.
        raw_true_xy = y_true[l][..., :2]*grid_shapes[l][::-1] - grid
        raw_true_wh = K.log(y_true[l][..., 2:4] / anchors[anchor_mask[l]] * input_shape[::-1])
        raw_true_wh = K.switch(object_mask, raw_true_wh, K.zeros_like(raw_true_wh)) # avoid log(0)=-inf
        box_loss_scale = 2 - y_true[l][...,2:3]*y_true[l][...,3:4]

        # Find ignore mask, iterate over each of batch.
        ignore_mask = tf.TensorArray(K.dtype(y_true[0]), size=1, dynamic_size=True)
        object_mask_bool = K.cast(object_mask, 'bool')
        def loop_body(b, ignore_mask):
            true_box = tf.boolean_mask(y_true[l][b,...,0:4], object_mask_bool[b,...,0])
            iou = box_iou(pred_box[b], true_box)
            best_iou = K.max(iou, axis=-1)
            try:
                tf.print("@@@train:",np.max(best_iou.numpy()),np.average(best_iou.numpy()))
            except:
                print()
            ignore_mask = ignore_mask.write(b, K.cast(best_iou<ignore_thresh, K.dtype(true_box)))
            return b+1, ignore_mask
        
        _, ignore_mask = tf.while_loop(lambda b,*args: b<m, loop_body, [0, ignore_mask])
        ignore_mask = ignore_mask.stack()
        ignore_mask = K.expand_dims(ignore_mask, -1)

        # K.binary_crossentropy is helpful to avoid exp overflow.
        xy_loss = object_mask * box_loss_scale * K.binary_crossentropy(raw_true_xy, raw_pred[...,0:2], from_logits=True)
        wh_loss = object_mask * box_loss_scale * 0.5 * K.square(raw_true_wh-raw_pred[...,2:4])
        confidence_loss = obj_scale * object_mask * K.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True)+ \
            noobj_scale * (1-object_mask) * K.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True) * ignore_mask
        class_loss = object_mask * K.binary_crossentropy(true_class_probs, raw_pred[...,5:], from_logits=True)

        xy_loss = K.sum(xy_loss) / mf
        wh_loss = K.sum(wh_loss) / mf
        confidence_loss = K.sum(confidence_loss) / mf
        class_loss = K.sum(class_loss) / mf
        loss += xy_loss + wh_loss + confidence_loss + class_loss
        if print_loss:
            tf.print("\n-----loss:",l,xy_loss, wh_loss, confidence_loss, class_loss, K.max(K.sigmoid(raw_pred[...,4:5])))
            #tf.print('\n---\n')
    return loss

def yolo_loss_one(args, anchors, num_classes, ignore_thresh=.5, print_loss=False, obj_scale=1,noobj_scale=1,divider=32):
    '''Return yolo_loss tensor

    Parameters
    ----------
    yolo_outputs: list of tensor, the output of yolo_body or tiny_yolo_body
    y_true: list of array, the output of preprocess_true_boxes
    anchors: array, shape=(N, 2), wh
    num_classes: integer
    ignore_thresh: float, the iou threshold whether to ignore object confidence loss

    Returns
    -------
    loss: tensor, shape=(1,)

    '''
    num_layers = len(anchors)//3 # default setting
    yolo_outputs = args[:num_layers]
    y_true = args[num_layers:]
    
    anchor_mask = [[0,1,2]]
    input_shape = K.cast(K.shape(yolo_outputs[0])[1:3] * divider, K.dtype(y_true[0]))
    grid_shapes = K.cast(K.shape(yolo_outputs[0])[1:3], K.dtype(y_true[0]))
    m = K.shape(yolo_outputs[0])[0] # batch size, tensor
    mf = K.cast(m, K.dtype(yolo_outputs[0]))

    
    loss = 0
    
    y_true = y_true[0]
    object_mask = y_true[..., 4:5]
    true_class_probs = y_true[..., 5:]

    grid, raw_pred, pred_xy, pred_wh = yolo_head(yolo_outputs[0],
            anchors, num_classes, input_shape, calc_loss=True)
    pred_box = K.concatenate([pred_xy, pred_wh])

    # Darknet raw box to calculate loss.
    raw_true_xy = y_true[..., :2]*grid_shapes[::-1] - grid
    raw_true_wh = K.log(y_true[..., 2:4] / anchors * input_shape[::-1])
    raw_true_wh = K.switch(object_mask, raw_true_wh, K.zeros_like(raw_true_wh)) # avoid log(0)=-inf
    box_loss_scale = 2 - y_true[...,2:3]*y_true[...,3:4]

    # Find ignore mask, iterate over each of batch.
    ignore_mask = tf.TensorArray(K.dtype(y_true), size=1, dynamic_size=True)
    object_mask_bool = K.cast(object_mask, 'bool')
    def loop_body(b, ignore_mask):
        true_box = tf.boolean_mask(y_true[b,...,0:4], object_mask_bool[b,...,0])
        iou = box_iou(pred_box[b], true_box)
        best_iou = K.max(iou, axis=-1)
        try:
            tf.print("@@@train:",np.max(best_iou.numpy()),np.average(best_iou.numpy()))
        except:
            print()
        ignore_mask = ignore_mask.write(b, K.cast(best_iou<ignore_thresh, K.dtype(true_box)))
        return b+1, ignore_mask
    
    _, ignore_mask = tf.while_loop(lambda b,*args: b<m, loop_body, [0, ignore_mask])
    ignore_mask = ignore_mask.stack()
    ignore_mask = K.expand_dims(ignore_mask, -1)

    # K.binary_crossentropy is helpful to avoid exp overflow.
    xy_loss = object_mask * box_loss_scale * K.binary_crossentropy(raw_true_xy, raw_pred[...,0:2], from_logits=True)
    wh_loss = object_mask * box_loss_scale * 0.5 * K.square(raw_true_wh-raw_pred[...,2:4])
    confidence_loss = obj_scale * object_mask * K.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True)+ \
        noobj_scale * (1-object_mask) * K.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True) * ignore_mask
    class_loss = object_mask * K.binary_crossentropy(true_class_probs, raw_pred[...,5:], from_logits=True)

    xy_loss = K.sum(xy_loss) / mf
    wh_loss = K.sum(wh_loss) / mf
    confidence_loss = K.sum(confidence_loss) / mf
    class_loss = K.sum(class_loss) / mf
    loss += xy_loss + wh_loss + confidence_loss + class_loss
    if print_loss:
        tf.print("\n-----loss:",xy_loss, wh_loss, confidence_loss, class_loss, K.max(K.sigmoid(raw_pred[...,4:5])))
        #tf.print('\n---\n')

    return loss

def yolo_loss_iou_one(args, anchors, num_classes, ignore_thresh=.5, print_loss=False, obj_scale=1,noobj_scale=1,divider=32,iou_type='siou'):
    '''Return yolo_loss tensor

    Parameters
    ----------
    yolo_outputs: list of tensor, the output of yolo_body or tiny_yolo_body
    y_true: list of array, the output of preprocess_true_boxes
    anchors: array, shape=(N, 2), wh
    num_classes: integer
    ignore_thresh: float, the iou threshold whether to ignore object confidence loss

    Returns
    -------
    loss: tensor, shape=(1,)

    '''
    balance         = [0.4, 1.0, 4], 
    box_ratio       = 0.05, 
    obj_ratio       = 1
    cls_ratio       = 0.5 / 4,

    num_layers = len(anchors)//3 # default setting
    yolo_outputs = args[:num_layers]
    y_true = args[num_layers:]
    
    anchor_mask = [[0,1,2]]
    input_shape = K.cast(K.shape(yolo_outputs[0])[1:3] * divider, K.dtype(y_true[0]))
    grid_shapes = K.cast(K.shape(yolo_outputs[0])[1:3], K.dtype(y_true[0]))
    m = K.shape(yolo_outputs[0])[0] # batch size, tensor
    mf = K.cast(m, K.dtype(yolo_outputs[0]))

    
    loss = 0
    
    y_true = y_true[0]
    object_mask = y_true[..., 4:5]
    true_class_probs = y_true[..., 5:]

    grid, raw_pred, pred_xy, pred_wh = yolo_head(yolo_outputs[0],
            anchors, num_classes, input_shape, calc_loss=True)
    pred_box = K.concatenate([pred_xy, pred_wh])

    # Darknet raw box to calculate loss.
    raw_true_xy = y_true[..., :2]*grid_shapes[::-1] - grid
    raw_true_wh = K.log(y_true[..., 2:4] / anchors * input_shape[::-1])
    raw_true_wh = K.switch(object_mask, raw_true_wh, K.zeros_like(raw_true_wh)) # avoid log(0)=-inf
    box_loss_scale = 2 - y_true[...,2:3]*y_true[...,3:4]

    # Find ignore mask, iterate over each of batch.
    ignore_mask = tf.TensorArray(K.dtype(y_true), size=1, dynamic_size=True)
    object_mask_bool = K.cast(object_mask, 'bool')
    def loop_body(b, ignore_mask):
        true_box = tf.boolean_mask(y_true[b,...,0:4], object_mask_bool[b,...,0])
        iou = box_iou(pred_box[b], true_box)
        best_iou = K.max(iou, axis=-1)
        try:
            tf.print("@@@train:",np.max(best_iou.numpy()),np.average(best_iou.numpy()))
        except:
            print()
        ignore_mask = ignore_mask.write(b, K.cast(best_iou<ignore_thresh, K.dtype(true_box)))
        return b+1, ignore_mask
    
    _, ignore_mask = tf.while_loop(lambda b,*args: b<m, loop_body, [0, ignore_mask])
    ignore_mask = ignore_mask.stack()
    ignore_mask = K.expand_dims(ignore_mask, -1)

    # K.binary_crossentropy is helpful to avoid exp overflow.
    raw_true_box    = y_true[...,0:4]
    iou             = box_iou_loss(pred_box, raw_true_box, iou_type)
    iou_loss        = object_mask * (1 - iou)
    location_loss   = K.sum(iou_loss)

    confidence_loss = obj_scale * object_mask * K.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True)+ \
        noobj_scale * (1-object_mask) * K.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True) * ignore_mask
    class_loss = object_mask * K.binary_crossentropy(true_class_probs, raw_pred[...,5:], from_logits=True)

    #-----------------------------------------------------------#
    #   计算正样本数量
    #-----------------------------------------------------------#
    num_pos         = tf.maximum(K.sum(K.cast(object_mask, tf.float32)), 1)
    num_neg         = tf.maximum(K.sum(K.cast((1 - object_mask) * ignore_mask, tf.float32)), 1)

    #-----------------------------------------------------------#
    #   将所有损失求和
    #-----------------------------------------------------------#
    location_loss   = location_loss * box_ratio / num_pos
    confidence_loss = K.sum(confidence_loss) * balance[0] * obj_ratio / (num_pos + num_neg)
    class_loss      = K.sum(class_loss) * cls_ratio / num_pos / num_classes

    loss            += location_loss + confidence_loss + class_loss
    if print_loss:
        tf.print("\n-----loss:",location_loss, confidence_loss, class_loss, K.max(K.sigmoid(raw_pred[...,4:5])))
        #tf.print('\n---\n')

    return loss

from utils import yolo_cfg, get_random_data
if __name__ == '__main__':
    
    input_shape = (112,112,3)
    m = tiny_yolo_res_body(input_shape,3,1,1,[8])
    m.summary()
    '''
    m.save('112.h5')

    converter = tf.lite.TFLiteConverter.from_keras_model(m)
    converter.experimental_new_converter = False
    converter.experimental_new_quantizer = True

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    #converter.target_spec.supportes_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.float32
    tflite_model_quant = converter.convert() 
    m_path = '112.tflite'
    with open(m_path,'wb') as f:
        f.write(tflite_model_quant)
        f.close()
    '''
    input_shape = (112,112,3)
    m = tiny_yolo_res_body(input_shape,3,1,1,[8])
    m.summary()
    m.save('112.h5')

    converter = tf.lite.TFLiteConverter.from_keras_model(m)

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supportes_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.float32

    cfg = yolo_cfg()
    annotation_path = cfg.train_data
    cfg.width = 112
    cfg.height = 112
    with open(annotation_path) as f:
        annotation_lines = f.readlines()
    annotation_lines = annotation_lines[0:500]
    def representative_data_gen():
        '''data generator for fit_generator'''
        n = len(annotation_lines)
        i = 0
        input_shape = (cfg.width,cfg.height)
        
        image_data = []
        box_data = []
        for b in range(100):
            if i==0:
                np.random.shuffle(annotation_lines)
            image, box = get_random_data(annotation_lines[i], input_shape, random=True)
            image = (image).astype('float32')
            image_data.append(image)
            box_data.append(box)
            i = (i+1) % n
        #image_data = np.array(image_data).astype('int8')
        image_data = np.array(image_data)
        image_data = image_data.reshape(100,cfg.width,cfg.height,3)
        for input_value in image_data:
            input_value = input_value.reshape(1,cfg.width,cfg.height,3)
            yield [input_value]

    converter.representative_dataset = representative_data_gen
    tflite_model_quant = converter.convert() 
    m_path = '112.tflite'
    with open(m_path,'wb') as f:
        f.write(tflite_model_quant)
        f.close()

    print('done')

