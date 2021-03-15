import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
tf.get_logger().setLevel('INFO')

import numpy as np
import matplotlib.image as mpimg
import json

def get_image(fpath, input_shape = None): 

    # image path
    if tf.is_tensor(fpath): fpath = fpath.numpy().decode('utf-8') # decode
    image_path = fpath.format('img') # '.../{}/000000.png' → '.../img/000000.png'

    # load image
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=None, interpolation='nearest')
    image = np.array(image).astype('uint8')
    image = tf.convert_to_tensor(image)

    # resize
    if input_shape is not None: 
        image = tf.image.resize(image, size = input_shape, method = 'nearest')

    return image

def get_seg(fpath, input_shape = None):

    # seg path
    if tf.is_tensor(fpath): fpath = fpath.numpy().decode('utf-8') # decode
    seg_path = fpath.format('instance') # '.../{}/000000.png' → '.../seg/000000.png'

    # load seg
    seg = mpimg.imread(seg_path)
    seg = seg * 255.0
    seg = seg[...,:3].astype('uint8')

    # resize
    if input_shape is not None: 
        seg = tf.image.resize(seg, size = input_shape, method = 'nearest')

    return seg

def create_masks(fpath, input_shape, str_label2clr, include_beacon = False, min_pixels = 20):

    # string to dict
    if tf.is_tensor(str_label2clr): str_label2clr = str_label2clr.numpy().decode('utf-8') 
    label2clr = json.loads(str_label2clr)

    labels = ['def_line_1', 'def_line_2', 'def_line_3', 'def_line_4', 'tmp_line_1', 'tmp_line_2', 'tmp_line_3', 'tmp_line_4']

    clr_keys = [label2clr[label] for label in labels]

    seg = get_seg(fpath, input_shape)

    # include beacon color tags [255, 10, 35 + x]
    if include_beacon:
        clrs = np.unique(np.reshape(seg, newshape = (-1,3)), axis = 0)

        for clr in clrs:
            if np.all(clr[:2] == [255,10]): clr_keys += [clr]


    instance_mask = np.zeros(input_shape)
    seg_mask = np.zeros(input_shape)

    # instance mask
    for index, clr_key in enumerate(clr_keys):
        instance_mask = np.where(np.all(seg == clr_key, axis = -1), np.max(instance_mask) + 1, instance_mask)

    # roove instances with less then min_pixels
    for idx in np.unique(instance_mask):
        if len(instance_mask[instance_mask==idx]) < min_pixels:
            instance_mask[instance_mask==idx] = 0

    # segmentation mask
    classes = {1: ['def_line_1', 'def_line_2', 'def_line_3', 'def_line_4'],
               2: ['tmp_line_1', 'tmp_line_2', 'tmp_line_3', 'tmp_line_4'],
               3: ['chevron_sign']
            }

    for id, keys in classes.items():
        for key in keys: 
            seg_mask = np.where(np.all(seg == label2clr[key], axis = -1), id, seg_mask)
  
    seg_mask, instance_mask = tf.cast(seg_mask[...,None], tf.uint8), tf.cast(instance_mask[...,None], tf.uint8)

    return seg_mask, instance_mask


def data_augmentation(image, binary_mask, instance_mask, prob = 0.5):
    '''image (H,W,3), binary_mask (H,W,1), instance_mask (H,W,1)'''
    # flip left-right
    if tf.random.uniform(shape = (1,), minval = 0.0, maxval = 1.0) <= prob:
        image = tf.image.flip_left_right(image)
        binary_mask = tf.image.flip_left_right(binary_mask)
        instance_mask = tf.image.flip_left_right(instance_mask)
  
    return image, binary_mask, instance_mask

def tf_preprocess_data(fpath, str_label2clr, input_shape = (360,640), num_classes = 4, is_training = True):

    # READ IMAGE
    [image,] = tf.py_function(func = get_image, 
                              inp  = [fpath, input_shape], 
                              Tout = [tf.uint8])
    image.set_shape(input_shape + (3,))

    # CREATE MASK
    seg_mask, instance_mask = tf.py_function(func  = create_masks, 
                                              inp  = [fpath, input_shape, str_label2clr], 
                                              Tout = [tf.uint8, tf.uint8])
  
    seg_mask.set_shape(input_shape + (1,))
    instance_mask.set_shape(input_shape + (1,))

    # DATA AUGMENTATION
    if is_training:
        image, seg_mask, instance_mask = data_augmentation(image, seg_mask, instance_mask)

    # NORMALIZE
    image = tf.cast(image, tf.float32) / 255.0

    # ONE-HOT-ENCODING
    seg_mask = tf.one_hot(tf.squeeze(seg_mask, axis = -1), depth = num_classes, axis=-1)

    labels = {'seg_mask': seg_mask, 'instance_mask': instance_mask}

    return image, labels


def get_color_map():
    cityscapes_color_map = {
        'unlabeled'     :[  0,   0,   0],
        'bilding'       :[ 70,  70,  70],
        'fence'         :[100,  40,  40],
        'other'         :[ 55,  90,  80],
        'pedestrian'    :[220,  20,  60],
        'pole'          :[153, 153, 153],
        'road_line'     :[157, 234,  50],
        'road'          :[128,  64, 128],
        'sidewalk'      :[244,  35, 232], 
        'vegetation'    :[107, 142,  35],
        'vehicle'       :[  0,   0, 142], 
        'wall'          :[102, 102, 156], 
        'traffic_sign'  :[220, 220,   0], 
        'sky'           :[70,  130, 180], 
        'grond'         :[ 81,   0,  81], 
        'bridge'        :[150, 100, 100], 
        'rail_track'    :[230, 150, 140], 
        'guard_rail'    :[180, 165, 180], 
        'traffic_light' :[250, 170,  30], 
        'static'        :[110, 190, 160], 
        'dynamic'       :[170, 120,  50], 
        'water'         :[ 45,  60, 150], 
        'terrain'       :[145, 170, 100], 
        'def_line_1'    :[255, 255, 255], 
        'def_line_2'    :[235, 235, 235], 
        'def_line_3'    :[215, 215, 215], 
        'def_line_4'    :[195, 195, 195], 
        'tmp_line_1'    :[255, 255,   0], 
        'tmp_line_2'    :[235, 235,   0], 
        'tmp_line_3'    :[215, 215,   0], 
        'tmp_line_4'    :[195, 195,   0], 
        'chevron_sign'  :[255,  10,  35],  
    }

    return cityscapes_color_map
