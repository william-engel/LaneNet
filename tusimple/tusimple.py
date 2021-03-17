import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
tf.get_logger().setLevel('INFO')

import numpy as np
import json
import cv2
from PIL import Image
        
def load_image(image_dir, label_data):
    image_dir = image_dir.numpy().decode('utf-8') # to str
    
    label_data = json.loads(label_data.numpy())
    filepath =  os.path.join(image_dir, label_data['raw_file'])
    image = tf.io.read_file(filepath)
    image = tf.io.decode_jpeg(image, channels = 3)
    image = tf.cast(image, dtype = tf.uint8)
    return image

def create_masks(label_data, image):
    '''
    label_data = '{"lanes": [[],[],...], "h_samples": [], "raw_file": "image000.jpg"}'
    label_data is a dict in string format
    '''
    im_height, im_width, _ = image.shape

    label_data = label_data.numpy().decode('utf-8') # to str

    # create empty masks
    binary_mask   = np.zeros(shape = [im_height, im_width, 1], dtype = 'uint8')
    instance_mask = np.zeros(shape = [im_height, im_width, 1], dtype = 'uint8')

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

def random_rotate(image, binary_mask, instance_mask, min_deg = -10, max_deg = 10):
    '''image (H,W,3) [0, 255] uint8, binary_mask (H,W,1) [0, 255] uint8, instance_mask (H,W,1) [0, 255] uint8'''

    rand_angle = np.random.randint(min_deg, max_deg) # random angle

    images = [image, binary_mask[...,0], instance_mask[...,0]] # (H,W,1) → (H,W)
    images = [np.array(image) if tf.is_tensor(image) else image for image in images] # to numpy
    images = [Image.fromarray(image).rotate(rand_angle, resample = Image.NEAREST) for image in images] # to PIL → rotate
    images = [tf.convert_to_tensor(np.array(image)) for image in images] # to tensor
    
    return images[0], images[1][...,None], images[2][...,None] # (H,W) → (H,W,1)

def tf_random_rotate(image, binary_mask, instance_mask, prob = 0.5):

    if tf.random.uniform(shape = (1,), minval = 0.0, maxval = 1.0) <= prob:
        [image, binary_mask, instance_mask] = tf.py_function(func = random_rotate,
                                                            inp  = [image, binary_mask, instance_mask],
                                                            Tout = [tf.uint8, tf.uint8, tf.uint8])
        
        image.set_shape([None, None, 3])
        binary_mask.set_shape([None, None, 1])
        instance_mask.set_shape([None, None ,1])
    
    return image, binary_mask, instance_mask

def tf_random_flip(image, binary_mask, instance_mask, prob = 0.5):
    '''image (H,W,3), binary_mask (H,W,1), instance_mask (H,W,1)'''

    if tf.random.uniform(shape = (1,), minval = 0.0, maxval = 1.0) <= prob:
        image = tf.image.flip_left_right(image)
        binary_mask = tf.image.flip_left_right(binary_mask)
        instance_mask = tf.image.flip_left_right(instance_mask)

    return image, binary_mask, instance_mask

def tf_data_augmentation(image, binary_mask, instance_mask):
  
    image, binary_mask, instance_mask = tf_random_flip(image, binary_mask, instance_mask)
    image, binary_mask, instance_mask = tf_random_rotate(image, binary_mask, instance_mask)

    return image, binary_mask, instance_mask

def preprocess_data(json_label, image_dir, input_shape = (480,640), num_classes = 2, is_training = True):
    # READ IMAGE
    [image,] = tf.py_function(func = load_image, 
                              inp  = [image_dir, json_label], 
                              Tout = [tf.uint8])

    image.set_shape([None, None, 3])
    
    # CREATE MASK
    binary_mask, instance_mask = tf.py_function(func = create_masks, 
                                                inp  = [json_label, image], 
                                                Tout = [tf.uint8, tf.uint8])

    binary_mask.set_shape([None, None, 1])
    instance_mask.set_shape([None, None ,1])
    
    # DATA AUGMENTATION
    if is_training:
        image, binary_mask, instance_mask = tf_data_augmentation(image, binary_mask, instance_mask)

    # RESIZE
    binary_mask = tf.image.resize(binary_mask, input_shape, method = 'nearest')
    instance_mask = tf.image.resize(instance_mask, input_shape, method = 'nearest')
    image = tf.image.resize(image, input_shape, method = 'nearest')

    # NORMALIZE
    image = tf.cast(image, tf.float32) / 255.0

    # ONE-HOT-ENCODING
    binary_mask = tf.one_hot(tf.squeeze(binary_mask, axis = -1), depth = num_classes, axis=-1)

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

def json_to_stringlist(label_paths):
  '''Takes a list of .json files.'''
  for path in label_paths: assert(os.path.exists(path)), f'Path not found: {path}'

  label_data = []
  for path in label_paths:
    with open(path) as json_file:
      for line in json_file:
        label_data.append(line)

  return label_data


def fit_lanes2masks(masks, labels, input_shape, original_shape, H = None, degree = 3):

    width_inp2orig  = original_shape[1] / input_shape[1] # width scale factor
    height_inp2orig = original_shape[0] / input_shape[0] # height scale factor

    '''
    masks: (B,W,H,1)
    H: (B,3,3)
    returns:
    labels_true[{'lanes':[[]], ..}, {}, ..]
    labels_pred[{'lanes':[[]], ..}, {}, ..]
    '''
    labels_pred = []
    labels_true = []

    if H is None: H = np.eye(N = 3)

    for mask, label in zip(masks, labels):
        label = label.numpy().decode('utf-8')
        label = json.loads(label)

        h = np.array(label['h_samples']) 

        lanes = []
        for id in np.unique(mask):
            if id == 0: continue

            # instance pixel coordinates
            y, x = np.where(mask == id)

            # scale from input size to original size
            y = y * height_inp2orig
            x = x * width_inp2orig

            # mask all h_samples hat are out of range
            out_range = np.any([h < np.min(y), h > np.max(y)], axis = 0)

            # apply homography
            pts = np.stack([x, y], axis = 1) # Nx2
            pts = np.pad(pts, [[0,0],[0,1]], constant_values = 1).T # 3xN
            pts = np.matmul(H, pts)  # 3xN
            xn, yn = pts[:-1] / pts[-1] # N

            # apply homography
            pts = np.pad(h[...,None], [[0,0],[1,1]], constant_values = 1).T # 3xN
            pts = np.matmul(H, pts)  # 3xN
            hn = pts[1] / pts[-1] # N

            # fit polynomial
            c = np.polyfit(yn, xn, degree)
            p = np.poly1d(c)

            # calculate new x coordinates
            xn = np.polyval(p, hn)

            # reverse homography
            pts = np.stack([xn, hn], axis = 1) # Nx2
            pts = np.pad(pts, [[0,0],[0,1]], constant_values = 1).T # 3xN
            pts = np.matmul(np.linalg.inv(H), pts)
            xn = pts[0] / pts[-1] # N

            # remove coordinates that are out of range
            xn[out_range] = -2

            lanes.append(xn.tolist())

        label_pred = label.copy()
        label_pred['lanes'] = lanes

        labels_pred.append(label_pred)
        labels_true.append(label)

    return labels_true, labels_pred

def apply_homography(lanes, h_samples, H):
    '''
    lanes: (N,M)
    h_samples: (M,)
    H: (3,3)
    '''
    lanes_ = []
    for lane in lanes:
        pts = np.stack([lane, h_samples], axis = 1) # Nx2
        pts = np.pad(pts, [[0,0],[0,1]], constant_values = 1).T # 3xN
        pts = np.matmul(H, pts)  # 3xN
        xn, yn = pts[:-1] / pts[-1] # N
        xn[lane == -2] = -2
        lanes_ += [xn.astype('int')]
    
    h_samples_ = yn.astype('int')

    return lanes_, h_samples_

def plot_lanes_on_image(image, lanes, h_samples):
    for lane in lanes:
        coor = np.vstack([lane, h_samples]).T.astype('int32') 
        coor = coor[coor[:,0] != -2]
        cv2.polylines(image, pts = [coor], isClosed = False, color = [0,0,255], thickness = 4)

    return image / 255.0

def plot_labels_on_images(labels, images, input_shape, original_shape,  H_matrices = None):
    '''images: (B,W,H,3) [0.0,1.0]'''
    width_orig2inp  = input_shape[1] / original_shape[1] # width scale factor
    height_orig2inp = input_shape[0] / original_shape[0] # height scale factor

    B, Height, Width, _ = images.shape

    if H_matrices is None:
        H_matrices = np.repeat(np.eye(3)[None,...], B, axis = 0)

    images_with_lanes = []
    for image, label, H in zip(images, labels, H_matrices):
        if isinstance(label, str): label = json.loads(label)

        lanes       = np.array(label['lanes'])
        h_samples   = np.array(label['h_samples']) * height_orig2inp # scale y-coordinates
        lanes[lanes != -2] =  lanes[lanes != -2] * width_orig2inp    # scale x-coordinates

        image = np.array(image * 255.0).astype('uint8')
        lanes, h_samples = apply_homography(lanes, h_samples, H)
        image = cv2.warpPerspective(image, H, (Width, Height))


        image = plot_lanes_on_image(image, lanes, h_samples)

        images_with_lanes.append(image)

    return images_with_lanes

