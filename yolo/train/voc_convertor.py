import xml.etree.ElementTree as ET
import os
from os import getcwd
from utils import yolo_cfg
import numpy as np



def convert_annotation(voc_folder, image_id, list_file,classes):
    fname = image_id[0:image_id.rfind('.')]
    in_file = open('%s/Annotations/%s.xml'%(voc_folder, fname))
    tree=ET.parse(in_file)
    root = tree.getroot()

    jpg_file = "%s/JPEGImages/%s"%(voc_folder,image_id)

    obj_count = 0
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            #print('cls: %s not found in %s difficult=%d'%(cls,image_id,int(difficult)))
            continue
        obj_count = obj_count +1
    
    if obj_count == 0:
        print('no easy object in %s'%image_id)
        return

    list_file.write("%s"%jpg_file)
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            #print('cls: %s not found in %s difficult=%d'%(cls,image_id,int(difficult)))
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(eval(xmlbox.find('xmin').text)), int(eval(xmlbox.find('ymin').text)), int(eval(xmlbox.find('xmax').text)), int(eval(xmlbox.find('ymax').text)))
        c = []
        for a in b:
            if a < 0:
                c.append(0)
            else:
                c.append(a)
        list_file.write("$" + ",".join([str(a) for a in c]) + ',' + str(cls_id))
    list_file.write('\n')

def get_images(path):
    list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if os.path.splitext(file)[1] == '.jpeg' or os.path.splitext(file)[1] == '.jpg':
                list.append(file)
    return list

def voc_convertor(voc_folder, train_txt,eva_txt,voc_classes):
    
    print('\n----voc_convertor_start----')
    img_list = get_images("%s/JPEGImages"%voc_folder)
    if len(img_list) == 0:
        print('\n----voc_convertor_error----')
        print('----please_cheak_img_folder----\n')
        return
    val_split = 0.1
    np.random.seed(10101)
    np.random.shuffle(img_list)
    np.random.seed(None)
    num_val = int(len(img_list)*val_split)
    num_train = len(img_list) - num_val
    train_list = img_list[:num_train]
    eva_list = img_list[num_train:-1]

    train_file = open(train_txt,'w')
    for img in train_list:
        convert_annotation(voc_folder,img,train_file,voc_classes)
    train_file.close()

    eva_file = open(eva_txt,'w')
    for img in eva_list:
        convert_annotation(voc_folder,img,eva_file,voc_classes)
    eva_file.close()
    print('\n----voc_convertor_finish----\n')
if __name__ == '__main__':
    cfg = yolo_cfg()
    voc_classes = cfg.class_names
    voc_convertor(cfg.voc_folder,cfg.train_data,cfg.eva_data,voc_classes)