import tensorflow as tf 
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input
import numpy as np 
import os 
from utils import yolo_cfg
from model_resnet import yolo_eval
import cv2
import argparse
from PIL import Image,ImageDraw
from evaluate import get_yolo_boxes

def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 6)

if __name__ == '__main__':
    parser = argparse.ArgumentParser() 
    parser.add_argument('-model', help='trained tflite', default=r'./yolo3_iou_smartcar_final.tflite',type=str)
    parser.add_argument('-image', help='test image', default=r'./test.jpg',type=str)
    args, unknown = parser.parse_known_args()

    labels = ['person']
    cfg = yolo_cfg()
    anchors_path = cfg.cluster_anchor
    anchors = get_anchors(anchors_path)
    anchors_num = len(anchors) / 3


    image = Image.open(args.image)
    origin_image = image
    image_w = image.size[0]
    image_h = image.size[1]
    
    scale = min(cfg.width/image_w, cfg.height/image_h)
    nw = int(image_w*scale)
    nh = int(image_h*scale)
    dx = (cfg.width-nw)//2
    dy = (cfg.height-nh)//2
    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', (cfg.width,cfg.height), (128,128,128))
    new_image.paste(image, (dx, dy))
    new_image.save('new_image.jpg')

    
    image_data = np.array(new_image)/255.
    image_data = image_data.reshape(1,cfg.width,cfg.height,3)


    pred_boxes,scores,classes = get_yolo_boxes(args.model,anchors,image_data,(image_w,image_h),0.12,0.5)

    im = origin_image
    draw = ImageDraw.Draw(im)
    for box in pred_boxes:
        draw.rectangle([box[0],box[1],box[2],box[3]],outline=tuple(np.random.randint(0, 255, size=[3])), width=2)
    im.save('tflite_detected_img.jpg')
    im.show()
    print('output tflite_detected_img.jpg ')



