import os
import xml.etree.ElementTree as ET

import numpy as np

from utils import general

names_to_idx = {
    0: 'aeroplane',
    1: 'bicycle',
    2: 'bird',
    3: 'boat',
    4: 'bottle',
    5: 'bus',
    6: 'car',
    7: 'cat',
    8: 'chair',
    9: 'cow',
    10: 'diningtable',
    11: 'dog',
    12: 'horse',
    13: 'motorbike',
    14: 'person',
    15: 'pottedplant',
    16: 'sheep',
    17: 'sofa',
    18: 'train',
    19: 'tvmonitor,'
}

names_to_idx = {v: k for k, v in names_to_idx.items()}
def parse_annotations_and_return_true_boxes(annotation_path, img):
    tree = ET.parse(annotation_path)
    root = tree.getroot()
    objs = []
    for obj in root.findall('object'):
        obj_dict = {}
        obj_dict['name'] = names_to_idx[obj.find('name').text]
        bbox = obj.find('bndbox')
        obj_dict['bbox'] = general.xyxy2xywhn(x=np.asarray([
            float(bbox.find('xmin').text),
            float(bbox.find('ymin').text),
            float(bbox.find('xmax').text),
            float(bbox.find('ymax').text),
        ]), w=img.shape[1], h=img.shape[0])
        objs.append(obj_dict)
    return objs
