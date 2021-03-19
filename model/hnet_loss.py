import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
tf.get_logger().setLevel('INFO')

import json
import numpy as np

def tf_reshape_H(coeff):
    coeffcient_slice = tf.concat([coeff, [1.0]], axis=-1) # (1, 6) -> (7,)
    H_indices = tf.constant([[0], [1], [2], [4], [5], [7], [8]]) 
    H_shape = tf.constant([9])
    H = tf.scatter_nd(H_indices, coeffcient_slice, H_shape)
    H = tf.reshape(H, shape=[3, 3])
    return H

def tf_apply_homography(x,y, H):

    # to homogeneous
    w = tf.ones_like(x)
    pts = tf.cast(tf.stack([x, y, w], axis = 0), tf.float32)

    # wrap perspective
    pts = tf.matmul(H, pts)

    # from homogeneous to cartesian
    yn = tf.transpose(pts[1, :] / pts[2, :])
    xn = tf.transpose(pts[0, :] / pts[2, :])

  return xn, yn

def read_str_label(str_label):
    if tf.is_tensor(str_label): str_label = str_label.numpy() # decode
    if isinstance(str_label, np.ndarray): str_label = str_label[0]

    label = json.loads(str_label)
    lanes = label['lanes']
    h_samples = label['h_samples']

    lanes_filtered = []
    for lane in lanes:
      if np.any(np.array(lane) > 0): lanes_filtered += [lane]

    return tf.cast(lanes_filtered, tf.float32), tf.cast(h_samples, tf.float32)

def tf_read_str_label(str_label):
    [lanes, h_samples] = tf.py_function(func = read_str_label,
                                        inp = [str_label],
                                        Tout = [tf.float32, tf.float32])
    lanes.set_shape([None, None])
    h_samples.set_shape([None,])
    return lanes, h_samples


            
def tf_polyfit_3(x,y):
    x, y = tf.cast(x, tf.float32), tf.cast(y, tf.float32)

    # [x³ x² x 1]
    ones = tf.ones_like(x, dtype=tf.float32)
    X = tf.stack([tf.pow(x, 3), tf.pow(x, 2), x, ones], axis=1)
    
    # p = (XᵀX)⁻¹Xᵀy
    p = tf.matmul(tf.matmul(tf.linalg.inv(tf.matmul(tf.transpose(X), X)), tf.transpose(X)), tf.expand_dims(y, -1))

    return p

def tf_polyval_3(p, x):
    p, x = tf.cast(p, tf.float32), tf.cast(x, tf.float32)

    # [x³ x² x 1]
    ones = tf.ones_like(x, dtype=tf.float32)
    X = tf.stack([tf.pow(x, 3), tf.pow(x, 2), x, ones], axis=1)

    y = tf.matmul(X, p) # (N,1)
    y = tf.squeeze(y, -1) # (N,)

    return y


def hnet_loss(str_labels, H_matrices):

    output_loss = tf.TensorArray(dtype = tf.float32, size = 0, dynamic_size = True)

    # batch size
    batch_size = tf.shape(H_matrices)[0]

    def outer_cond(i, H_matrices, output_loss):
        return tf.less(i, batch_size) 
    
    def outer_body(i, H_matrices, output_loss):

        # read string label
        lanes, h_samples = tf_read_str_label(str_labels[i])

        # number of lanes
        num_lanes = tf.shape(lanes)[0]

        # reshape homography
        H = tf_reshape_H(H_matrices[i])

        def inner_cond(j, lanes, output_loss):
            return tf.less(j, num_lanes) 
        
        def inner_body(j, lanes, output_loss):
          
            lane = lanes[j]

            # remove masked elements
            x = lane[lane != -2]
            y = h_samples[lane != -2]

            # apply homography
            xn, yn = tf_apply_homography(x,y,H)

            # the closed loop polyfit performs bad if the values are far away form origin
            yn = yn - tf.reduce_mean(yn) # shift to origin

            # polyfit
            p = tf_polyfit_3(yn, xn)

            # calculate new x-coordinates
            xn = tf_polyval_3(p, yn)

            # apply inverse homography
            xn, _ = tf_apply_homography(xn, yn, tf.linalg.inv(H))

            # calculate MSE
            loss = tf.reduce_mean(tf.pow(x - xn, 2))

            # append to total loss
            output_loss = output_loss.write(j*(i+1), loss)

            return j+1, lanes, output_loss

        _,_,output_loss = tf.while_loop(inner_cond, inner_body, [0, lanes, output_loss])

        return i+1, H_matrices, output_loss
    
    _,_,losses = tf.while_loop(outer_cond, outer_body, [0, H_matrices, output_loss])

    losses = losses.stack()
    loss = tf.reduce_mean(losses)

    return loss