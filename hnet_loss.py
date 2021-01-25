import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
tf.get_logger().setLevel('INFO')

import json
import numpy as np

def tf_json_to_tensor(json_label):
    def json_to_tensor(json_label):
      if isinstance(json_label, type(tf.constant(0))): json_label = json_label.numpy()[0]
      json_label = json.loads(json_label)
      lanes = tf.cast(json_label['lanes'], dtype = tf.float32)
      h_samples = tf.cast(json_label['h_samples'], dtype = tf.float32)
      return lanes, h_samples

    [lanes, h_samples] = tf.py_function(func = json_to_tensor, 
                                        inp  = [json_label], 
                                        Tout = [tf.float32, tf.float32])

    lanes.set_shape([None, None])
    h_samples.set_shape([None,])

    return lanes, h_samples

def tf_reshape_H(H):
    H = tf.cast(H, tf.float32)
    H = tf.concat([H, [1.0]], axis = 0) # (1,6) → (7,)
    H_indices = tf.constant([0,1,2,4,5,7,8], shape = (7,1))
    H = tf.scatter_nd(H_indices, H, shape = (9,))
    H = tf.reshape(H, shape = (3,3))
    return tf.cast(H, tf.float32)

def tf_apply_H(x, y, H):

    pts = tf.stack([x,y], axis = 0) # 2xN
    pts = tf.pad(pts, [[0,1],[0,0]], constant_values=1) #3xN
    pts = tf.cast(pts, tf.float32)

    pts = tf.matmul(H, pts) # 3xN

    xn = tf.transpose(pts[0] / pts[-1]) #N
    yn = tf.transpose(pts[1] / pts[-1]) #N

    return xn, yn

def tf_reverse_H(x, y, H):
    pts = tf.stack([x,y], axis = 0) # 2xN
    pts = tf.pad(pts, [[0,1],[0,0]], constant_values=1) #3xN
    pts = tf.cast(pts, tf.float32)

    pts = tf.matmul(tf.linalg.inv(H), pts) # 3xN

    xn = tf.transpose(pts[0] / pts[-1]) #N
    yn = tf.transpose(pts[1] / pts[-1]) #N

    return xn, yn


def hnet_loss(json_labels, H_matrices):

  output_loss = tf.TensorArray(dtype = tf.float32, size = 0, dynamic_size = True)

  def outer_cond(i, H_matrices, output_loss):
    return i < tf.shape(H_matrices)[0] # loop over batches
  
  def outer_body(i, H_matrices, output_loss):

    # read json label
    lanes, h_samples = tf_json_to_tensor(json_labels[i])


    def inner_cond(j, lanes, output_loss):
      return j < tf.shape(lanes)[0] # loop over lanes
    
    def inner_body(j, lanes, output_loss):
      
      lane = lanes[j]
      x = lane[lane != -2]
      y = h_samples[lane != -2]

      if tf.size(x) > 1:

        # reshape H matrix
        H = tf_reshape_H(H_matrices[i]) # 3x3

        # apply H matrix
        xn, yn = tf_apply_H(x, y, H)

        # polyfit
        xn = tf.numpy_function(func = lambda x,y: np.polyval(np.polyfit(y, x, 3), y),
                              inp  = [xn, yn],
                              Tout = [tf.float32])
        xn.set_shape([None,])

        # reverse H
        xn, yn = tf_reverse_H(xn, yn, H)

        # calculate loss 
        loss = tf.reduce_mean(tf.pow(x - xn, 2))

        output_loss = output_loss.write(j*(i+1), loss)

      return j+1, lanes, output_loss


    
    _,_,output_loss = tf.while_loop(inner_cond, inner_body, [0, lanes, output_loss])


    return i+1, H_matrices, output_loss
  
  _,_,losses = tf.while_loop(outer_cond, outer_body, [0, H_matrices, output_loss])

  losses = losses.stack()
  loss = tf.reduce_mean(losses)

  return loss