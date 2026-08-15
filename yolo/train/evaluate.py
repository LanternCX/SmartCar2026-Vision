import tensorflow as tf 
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input
from tensorflow.keras import backend as K
import numpy as np 
import os 
from utils import yolo_cfg
from model_resnet import yolo_eval
import cv2
import argparse
from PIL import Image,ImageDraw

from utils import get_random_data,yolo_cfg
from calc_mAP import get_map
def _sigmoid(x):
    return 1. / (1. + np.exp(-x))

def decode_output(netout,input_shape,image_shape,anchors,conf_thres):
    grid_h, grid_w = netout.shape[:2]
    nb_box = 3
    netout = netout.reshape((grid_h, grid_w, nb_box, -1))
    nb_class = netout.shape[-1] - 5
    net_w,net_h = input_shape
    image_w,image_h = image_shape
    boxes = []


    scores = netout[..., 4:5]
    classes = netout[..., 5:]
    scores = _sigmoid(scores)
    classes = _sigmoid(classes)

    netout[..., :2]  = _sigmoid(netout[..., :2])
    netout[..., 4:]  = _sigmoid(netout[..., 4:])
    #netout[..., 5:]  = netout[..., 4][..., np.newaxis] * netout[..., 5:]
    
    classes = netout[..., 5:]

    if (float(net_w)/image_w) < (float(net_h)/image_h):
        new_w = net_w
        new_h = (image_h*net_w)/image_w
    else:
        new_h = net_w
        new_w = (image_w*net_h)/image_h
    
    x_offset, x_scale = (net_w - new_w)/2./net_w, float(net_w)/new_w
    y_offset, y_scale = (net_h - new_h)/2./net_h, float(net_h)/new_h

    for i in range(grid_h*grid_w):
        row = int(i / grid_w)
        col = i % grid_w
        
        for b in range(nb_box):
            # 4th element is objectness score
            objectness = netout[int(row)][int(col)][b][4]
            #objectness = netout[..., :4]
            
            if(objectness <= conf_thres): continue
            
            # first 4 elements are x, y, w, and h
            x, y, w, h = netout[int(row)][int(col)][b][:4]

            x = (col + x) / grid_w # center position, unit: image width
            y = (row + y) / grid_h # center position, unit: image height 
            w = anchors[2 * b + 0] * np.exp(w) / net_w # unit: image width
            h = anchors[2 * b + 1] * np.exp(h) / net_h # unit: image height  
            
            # last elements are class probabilities
            classes = netout[int(row)][col][b][5:]
            
            x = (x - x_offset) * x_scale
            y = (y - y_offset) * y_scale
            w *= x_scale
            h *= y_scale

            box = ((x-w/2)*image_w, (y-h/2)*image_h, (x+w/2)*image_w, (y+h/2)*image_h, objectness, classes)
            if (box[0] > 0) and (box[0] < box[2]) and (box[2] < image_w) and (box[1] > 0) and (box[1]< box[3]) and (box[3] < image_h):
                boxes.append(box)

    return boxes

def compute_overlap(a, b):
    """
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    Parameters
    ----------
    a: (N, 4) ndarray of float
    b: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = np.minimum(np.expand_dims(a[:, 2], axis=1), b[:, 2]) - np.maximum(np.expand_dims(a[:, 0], 1), b[:, 0])
    ih = np.minimum(np.expand_dims(a[:, 3], axis=1), b[:, 3]) - np.maximum(np.expand_dims(a[:, 1], 1), b[:, 1])

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    ua = np.expand_dims((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), axis=1) + area - iw * ih

    ua = np.maximum(ua, np.finfo(float).eps)

    intersection = iw * ih

    return intersection / ua  
    
def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap     

def do_nmx_tf(box,iou_thres,num_classes):
    box = box.astype('float32')
    boxes = box[...,0:4]
    box_scores = box[...,4:]

    boxes_ = []
    scores_ = []
    classes_ = []
    max_boxes_tensor = K.constant(20, dtype='int32')
    for c in range(num_classes):
        # TODO: use keras backend instead of tf.
        class_boxes = tf.convert_to_tensor(boxes)
        class_box_scores = box_scores[...,0:1] * box_scores[...,1:]
        class_box_scores = K.concatenate(class_box_scores, axis=0)
        #class_box_scores = tf.convert_to_tensor(class_box_scores)
        nms_index = tf.image.non_max_suppression(
            class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=iou_thres)
        class_boxes = K.gather(class_boxes, nms_index)
        class_box_scores = K.gather(class_box_scores, nms_index)
        classes = K.ones_like(class_box_scores, 'int32') * c
        boxes_.append(class_boxes)
        scores_.append(class_box_scores)
        classes_.append(classes)
    boxes_ = K.concatenate(boxes_, axis=0)
    scores_ = K.concatenate(scores_, axis=0)
    classes_ = K.concatenate(classes_, axis=0)

    return boxes_,scores_,classes_


def get_yolo_boxes(model,anchors,img_data,img_shape,conf_thres,iou_thres):
    interpreter = tf.lite.Interpreter(model_path=str(model))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()
    input_type = input_details['dtype']
    input_shape = input_details['shape']

    if input_type == np.uint8:
        test_data = (img_data * 255).astype('uint8')
    elif input_type == np.int8:
        test_data = (img_data*255-128).astype('int8')
    elif input_type == np.float32:
        test_data = img_data.astype('float32')

    

    test_data = test_data.reshape(input_shape)
    interpreter.set_tensor(input_details["index"], test_data)
    interpreter.invoke()
    
    pred_boxes = []
    for i in range(len(output_details)):
        output = interpreter.get_tensor(output_details[i]["index"])[0]
        net_shape = input_shape[1:3]
        if(output_details[i]['dtype'] == np.int8):
            zero_point = output_details[i]['quantization_parameters']['zero_points']
            scale = output_details[i]['quantization_parameters']['scales']
            output = ((output - zero_point)*scale).astype('float32')
        box = decode_output(output,net_shape,img_shape,anchors[i],conf_thres)
        pred_boxes += (box)
    if len(pred_boxes) > 0:
        pred_boxes,scores,classes = do_nmx_tf(np.array(pred_boxes),iou_thres,1)
        return np.array(pred_boxes),np.array(scores),np.array(classes)
    else:
        return [],[],[]
    

def del_file(filepath):
    del_list = os.listdir(filepath)
    for f in del_list:
        file_path = os.path.join(filepath, f)
        if os.path.isfile(file_path):
            os.remove(file_path)

def evaluate(model,input_shape,annotation_lines,anchors,num_classes,iou_threshold,class_names,map_out_path):

    pred_scores = []
    n = len(annotation_lines)
    
    
    if not os.path.exists(map_out_path):
        os.mkdir(map_out_path)
    
    if not os.path.exists(os.path.join(map_out_path,"ground-truth/")):
        os.mkdir(os.path.join(map_out_path,"ground-truth/"))

    if not os.path.exists(os.path.join(map_out_path,"detection-results/")):
        os.mkdir(os.path.join(map_out_path,"detection-results/"))

    del_file(os.path.join(map_out_path,"ground-truth/"))
    del_file(os.path.join(map_out_path,"detection-results/"))
    for i in range(n):
        if i==0:
            np.random.shuffle(annotation_lines)
        image, box = get_random_data(annotation_lines[i], input_shape, random=False)
        true_box = [b for b in box if b[0] > 0 and b[1] > 0]
        image_file = os.path.basename(annotation_lines[i].split('$')[0])
        image_id = image_file[0:image_file.rfind('.')]+'.txt'
        f = open(os.path.join(map_out_path,"ground-truth/"+image_id),'w')
        for b in true_box:
            f.write('%s %f %f %f %f\n'%(class_names[int(b[4])],b[0],b[1],b[2],b[3]))
            
        f.close()

        pred_boxes,pred_scores,pred_classes = get_yolo_boxes(model,anchors,image,input_shape,0.3,0.45)
        f = open(os.path.join(map_out_path,"detection-results/"+image_id),'w')
        if(len(pred_boxes) > 0):
            for i in range(len(pred_boxes)):
                f.write("%s %f %f %f %f %f\n"%(class_names[int(pred_classes[i])],pred_scores[i],pred_boxes[i][0],pred_boxes[i][1],pred_boxes[i][2],pred_boxes[i][3]))
        else:
            f.write("%s 0.0 0.0 0.0 0.0 0.0\n"%(class_names[int(0)]))
        
        f.close()
        

    return 0    


        
def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 6)

if __name__ == '__main__':
    cfg = yolo_cfg()
    tflites = ['yolo3_iou_smartcar_final.tflite']

    anchors = [get_anchors(cfg.cluster_anchor)]
    print('evaluate begin')
    with open(cfg.eva_data,'r') as f:
        lines = f.readlines()
        f.close()
    if not os.path.exists('./map_out'):
        os.mkdir('./map_out')
    for i in range(len(tflites)):
        model = tflites[i]
        interpreter = tf.lite.Interpreter(model_path=str(model))
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()[0]
        input_shape = input_details['shape']
        input_shape = input_shape[1:3]
        num_classes = cfg.num_classes
        anchor = anchors[i]
        map_out_path = './map_out/'+ model[0:model.rfind('.')] + '_out_map'
        evaluate(model,input_shape,lines,anchor,num_classes,0.45,cfg.class_names,map_out_path)
        map = get_map(MINOVERLAP=0.5,draw_plot=True,score_threhold = 0.3,path =map_out_path)
        print("%s mAP:%f"%(model,map))
        

    print('done')

