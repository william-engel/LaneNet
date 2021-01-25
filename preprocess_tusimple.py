import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
tf.get_logger().setLevel('INFO')

import numpy as np
import json
import cv2

        
def read_image(image_dir, label_data):
    image_dir = image_dir.numpy().decode('utf-8') # to str
    
    label_data = json.loads(label_data.numpy())
    filepath =  os.path.join(image_dir, label_data['raw_file'])
    image = tf.io.read_file(filepath)
    image = tf.io.decode_jpeg(image, channels = 3)
    image = tf.cast(image, dtype = tf.float32)
    return image

def create_masks(label_data, image_shape):
    '''
    label_data = '{"lanes": [[],[],...], "h_samples": [], "raw_file": "image000.jpg"}'
    label_data is a dict in string format
    '''
    im_height, im_width = image_shape

    label_data = label_data.numpy().decode('utf-8') # to str

    # create empty masks
    binary_mask   = np.zeros(shape = [im_height, im_width, 1])
    instance_mask = np.zeros(shape = [im_height, im_width, 1])

    # read json
    label_data = json.loads(label_data)
    lanes = np.array(label_data['lanes'])

    # plot lane lines on masks
    for index, lane in enumerate(lanes):
        coordinates = np.vstack([lane, label_data['h_samples']]).T
        coordinates = coordinates[coordinates[:,0] != -2]
        cv2.polylines(binary_mask, [coordinates], isClosed = False, color = [1], thickness = 15)
        cv2.polylines(instance_mask, [coordinates], isClosed = False, color = [index + 1], thickness = 15)
    return binary_mask, instance_mask

def preprocess_data(json_label, image_dir, input_shape = (480,640), original_shape = (720,1280)):
    # READ IMAGE
    [image,] = tf.py_function(func = read_image, 
                              inp  = [image_dir, json_label], 
                              Tout = [tf.float32])
    
    # CREATE MASK
    binary_mask, instance_mask = tf.py_function(func = create_masks, 
                                                inp  = [json_label, original_shape], 
                                                Tout = [tf.int32, tf.int32])

    binary_mask.set_shape(original_shape + (1,))
    instance_mask.set_shape(original_shape + (1,))
    image.set_shape(original_shape + (3,))

    # RESIZE
    binary_mask = tf.image.resize(binary_mask, input_shape, method = 'nearest')
    instance_mask = tf.image.resize(instance_mask, input_shape, method = 'nearest')
    image = tf.image.resize(image, input_shape, method = 'nearest')

    # NORMALIZE
    image = image / 255.0

    # ONE-HOT-ENCODING
    binary_mask = tf.one_hot(tf.squeeze(binary_mask, axis = -1), depth = 2, axis=-1)

    image = tf.squeeze(image)
    instance_mask = tf.squeeze(instance_mask)

    labels = {'binary_mask': binary_mask, 'instance_mask': instance_mask, 'json_label': json_label}
    return image, labels

def remove_labels(image, labels, to_remove = None):
    if to_remove == None:
        return image, labels
    else:
        for label in to_remove:
            labels.pop(label, None)
        return image, labels