import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras import backend as K
from keras.layers import Input,Lambda
from keras.models import Model
from tensorflow.keras.optimizers import Adam
from keras.callbacks import TensorBoard,ModelCheckpoint,ReduceLROnPlateau,EarlyStopping,LearningRateScheduler

from utils import get_random_data,yolo_cfg
from utils import get_lr_scheduler

import model_resnet
from tflite_add_post_processing import add_post_node
def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)

def create_res_tiny_model(input_shape, anchors, num_classes, load_pretrained=False, freeze_body=2,
            weights_path='model_data/tiny_yolo_weights.h5',
            iou_threshhold=0.5,obj_scale=1,noobj_scale=1,num_heads=2,divider=32,iou_type='siou'):
    '''create the training model, for Tiny YOLOv3'''
    K.clear_session() # get a new session
    h, w = input_shape
    image_input = Input(shape=(h, w, 3))
    
    num_anchors = len(anchors)
    if num_heads == 2:
        y_true = [Input(shape=(h//{0:divider[0], 1:divider[1]}[l], w//{0:divider[0], 1:divider[1]}[l], \
            num_anchors//2, num_classes+5)) for l in range(2)]
        model_body = model_resnet.tiny_yolo_res_body((h, w, 3), num_anchors//num_heads, num_classes,num_heads,divider)
        print('Create Tiny YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

        if iou_type == 'iou':
            model_loss = Lambda(model_resnet.yolo_loss, output_shape=(1,), name='yolo_loss',
            arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': iou_threshhold, 'obj_scale':obj_scale,'noobj_scale':noobj_scale})(
            [*model_body.output, *y_true])
        else:
            model_loss = Lambda(model_resnet.yolo_loss_iou, output_shape=(1,), name='yolo_loss',
                arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': iou_threshhold, 'obj_scale':obj_scale,'noobj_scale':noobj_scale,'iou_type':iou_type})(
                [*model_body.output, *y_true])
        model = Model([model_body.input, *y_true], model_loss)
    elif num_heads == 1:
        
        y_true = [Input(shape=(h//divider[0], w//divider[0], num_anchors, num_classes+5))]
        model_body = model_resnet.tiny_yolo_res_body((h, w, 3), num_anchors//num_heads, num_classes,num_heads,divider)
        print('Create Tiny YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

        if iou_type == 'iou':
            model_loss = Lambda(model_resnet.yolo_loss_one, output_shape=(1,), name='yolo_loss',
                arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': iou_threshhold, 'obj_scale':obj_scale,'noobj_scale':noobj_scale,'divider':divider[0]})(
                [model_body.output, *y_true])
        else:
            model_loss = Lambda(model_resnet.yolo_loss_iou_one, output_shape=(1,), name='yolo_loss',
                arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': iou_threshhold, 'obj_scale':obj_scale,'noobj_scale':noobj_scale,'divider':divider[0],'iou_type':iou_type})(
                [model_body.output, *y_true])
        model = Model([model_body.input, *y_true], model_loss)


    
    #model = Model([model_body.input, *y_true], [model_body.output,*y_true])
    return model,model_body

def preprocess_true_boxes(true_boxes, input_shape, anchors, num_classes,divider):
    '''Preprocess true boxes to training input format

    Parameters
    ----------
    true_boxes: array, shape=(m, T, 5)
        Absolute x_min, y_min, x_max, y_max, class_id relative to input_shape.
    input_shape: array-like, hw, multiples of 32
    anchors: array, shape=(N, 2), wh
    num_classes: integer

    Returns
    -------
    y_true: list of array, shape like yolo_outputs, xywh are reletive value

    '''
    assert (true_boxes[..., 4]<num_classes).all(), 'class id must be less than num_classes'
    num_layers = len(anchors)//3 # default setting
    if (num_layers >1):
        anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5],[0,1,2]]
    elif num_layers == 1: 
        anchor_mask = [[0,1,2]]

    true_boxes = np.array(true_boxes, dtype='float32')
    input_shape = np.array(input_shape, dtype='int32')
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) / 2.0
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]
    true_boxes[..., 0:2] = boxes_xy/input_shape[::-1]
    true_boxes[..., 2:4] = boxes_wh/input_shape[::-1]

    m = true_boxes.shape[0]
    if (num_layers == 3):
        grid_shapes = [input_shape//{0:32, 1:16, 2:8}[l] for l in range(num_layers)]
        y_true = [np.zeros((m,grid_shapes[l][0],grid_shapes[l][1],len(anchor_mask[l]),5+num_classes),
            dtype='float32') for l in range(num_layers)]
    elif num_layers == 2:
        grid_shapes = [input_shape//{0:divider[0], 1:divider[1]}[l] for l in range(num_layers)]
        y_true = [np.zeros((m,grid_shapes[l][0],grid_shapes[l][1],len(anchor_mask[l]),5+num_classes),
            dtype='float32') for l in range(num_layers)]
    elif num_layers == 1:
        grid_shapes = [input_shape//divider[0]]
        y_true = [np.zeros((m,grid_shapes[0][0],grid_shapes[0][1],len(anchor_mask[0]),5+num_classes),
        dtype='float32')]

    # Expand dim to apply broadcasting.
    anchors = np.expand_dims(anchors, 0)
    anchor_maxes = anchors / 2.
    anchor_mins = -anchor_maxes
    valid_mask = boxes_wh[..., 0]>0

    for b in range(m):
        # Discard zero rows.
        wh = boxes_wh[b, valid_mask[b]]
        if len(wh)==0: continue
        # Expand dim to apply broadcasting.
        wh = np.expand_dims(wh, -2)
        box_maxes = wh / 2.
        box_mins = -box_maxes

        intersect_mins = np.maximum(box_mins, anchor_mins)
        intersect_maxes = np.minimum(box_maxes, anchor_maxes)
        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        box_area = wh[..., 0] * wh[..., 1]
        anchor_area = anchors[..., 0] * anchors[..., 1]
        iou = intersect_area / (box_area + anchor_area - intersect_area)

        # Find best anchor for each true box
        best_anchor = np.argmax(iou, axis=-1)

        for t, n in enumerate(best_anchor):
            for l in range(num_layers):
                if n in anchor_mask[l]:
                    i = np.floor(true_boxes[b,t,0]*grid_shapes[l][1]).astype('int32')
                    j = np.floor(true_boxes[b,t,1]*grid_shapes[l][0]).astype('int32')
                    k = anchor_mask[l].index(n)
                    c = true_boxes[b,t, 4].astype('int32')
                    y_true[l][b, j, i, k, 0:4] = true_boxes[b,t, 0:4]
                    y_true[l][b, j, i, k, 4] = 1
                    y_true[l][b, j, i, k, 5+c] = 1

    return y_true

def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes,divider):
    '''data generator for fit_generator'''
    n = len(annotation_lines)
    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            if i==0:
                np.random.shuffle(annotation_lines)
            image, box = get_random_data(annotation_lines[i], input_shape, random=True)
            image_data.append(image)
            box_data.append(box)
            i = (i+1) % n
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes,divider)
        yield [image_data, *y_true], np.zeros(batch_size)


def data_generator_wrapper(annotation_lines, batch_size, input_shape, anchors, num_classes,divider):
    n = len(annotation_lines)
    if n==0 or batch_size<=0: return None
    return data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes,divider)

import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser() 
    parser.add_argument('-body', help='body type', default=r'mbv2',type=str)
    args, unknown = parser.parse_known_args()

    cfg = yolo_cfg()
    annotation_path = cfg.train_data
    log_dir = 'logs/000/'
   
    anchors_path = cfg.cluster_anchor
    class_names = cfg.class_names
    num_classes = cfg.num_classes
    anchors = get_anchors(anchors_path)
    divider = cfg.divider
    if isinstance(divider, int):
        divider = [divider]
    input_shape = (cfg.width,cfg.height) # multiple of 32, hw

    if args.body == 'mbv2':
        print("head num:%d"%cfg.num_heads)
        model,infer_model = create_res_tiny_model(
            input_shape=input_shape, 
            anchors=anchors, 
            num_classes=num_classes,
            iou_threshhold=cfg.iou_threshold,
            obj_scale=cfg.obj_scale,
            noobj_scale=cfg.noobj_scale,
            num_heads=cfg.num_heads,
            divider=divider,
            iou_type=cfg.iou_type
        )
        model_name = 'mbv2_%dhead_ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'%cfg.num_heads
    else:
        print('unsupported body type')
        exit(-1)

    model.compile(optimizer=Adam(lr=1e-3), loss={
            # use custom yolo_loss Lambda layer.
            'yolo_loss': lambda y_true, y_pred: y_pred})

    model.summary()
    
    print('Build Model')

    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(log_dir + model_name,
        monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=15, verbose=1)

    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val
    
    '''
    '''
    lr_decay_type = 'cos'
    lr_start = 1e-2
    lr_min = 1e-6
    batch_size = cfg.batch_size 
    nbs             = 32
    lr_limit_max    = 1e-3 
    lr_limit_min    = 3e-6 
    epochs = cfg.total_epochs
    Init_lr_fit     = min(max(batch_size / nbs * lr_start, lr_limit_min), lr_limit_max)
    Min_lr_fit      = min(max(batch_size / nbs * lr_min, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

    print('lr_start:%f,Min_lr_fit:%f'%(lr_start, Min_lr_fit))
    lr_scheduler_func = get_lr_scheduler(lr_decay_type, lr_start, Min_lr_fit, epochs)
    lr_scheduler    = LearningRateScheduler(lr_scheduler_func, verbose = 1)
    '''
    '''

    
    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
    model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes,divider),
        steps_per_epoch=max(1, num_train//batch_size),
        validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes,divider),
        validation_steps=max(1, num_val//batch_size),
        epochs=cfg.total_epochs,
        initial_epoch=0,
        callbacks=[logging, checkpoint, lr_scheduler, early_stopping])
    model.save_weights(log_dir + '%s_trained_weights_final.h5'%cfg.body_type)

    infer_model.load_weights(log_dir + '%s_trained_weights_final.h5'%cfg.body_type)
    converter = tf.lite.TFLiteConverter.from_keras_model(infer_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    def representative_data_gen():
        '''data generator for fit_generator'''
        n = len(lines)
        i = 0
        input_shape = (cfg.width,cfg.height)
        
        image_data = []
        box_data = []
        for b in range(100):
            if i==0:
                np.random.shuffle(lines)
            image, box = get_random_data(lines[i], input_shape, random=True)
            image = (image).astype('float32')
            image_data.append(image)
            box_data.append(box)
            i = (i+1) % n

        image_data = np.array(image_data)
        image_data = image_data.reshape(100,cfg.width,cfg.height,3)
        for input_value in image_data:
            input_value = input_value.reshape(1,cfg.width,cfg.height,3)
            yield [input_value]

    converter.representative_dataset = representative_data_gen
    tflite_model_quant = converter.convert() 
    m_path = 'yolo3_%s_smartcar_final.tflite'%(cfg.iou_type)
    with open(m_path,'wb') as f:
        f.write(tflite_model_quant)
        f.close()
    add_post_node(m_path)
    print('Training Complete')


    
