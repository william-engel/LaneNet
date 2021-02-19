import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
tf.get_logger().setLevel('INFO')

import time
import numpy as np

from sklearn.cluster import MeanShift, estimate_bandwidth


def create_instance_masks(embeddings, binary_masks, max_num_lanes = 5, min_pixels = 15, min_bin_freq = 1, bandwidth = None):

    B, H, W, E = embeddings.shape # (batch, height, width, embedding dim)
    time_taken = [] # stop time

    instance_masks = []
    for embedding, binary_mask in zip(embeddings, binary_masks):
        instance_mask = np.zeros([H, W], dtype=np.uint8) 

        embedding = embedding[binary_mask != 0] # remove backround

        if bandwidth == None:
            bandwidth = estimate_bandwidth(embedding, quantile=0.2, n_samples=100)
        
        ms = MeanShift(bandwidth = bandwidth, bin_seeding = True, min_bin_freq = min_bin_freq)

        if len(embedding) != 0:

            start_time = time.time() # start time
            labels = ms.fit(embedding).labels_
            time_taken += [time.time() - start_time] # stop time

            instance_mask[binary_mask != 0] = labels + 1

            # remove to small lane instances
            for id in np.unique(instance_mask):
                if len(instance_mask == id) < min_pixels:
                    instance_mask[instance_mask == id] = 0
            
            # remove all lanes aboves max_num_lanes
            ids, cnt = np.unique(instance_mask, return_counts=True)

            if len(ids) > max_num_lanes:
                sindex = np.argsort(cnt)
                for id in ids[sindex[max_num_lanes:]]:
                    instance_mask[instance_mask == id] = 0
    
        instance_masks.append(instance_mask)

    time_taken = np.mean(time_taken)
    instance_masks = np.asarray(instance_masks)

    return instance_masks, time_taken


def reshape_H(H_matrices):
    '''H_matracies (B,6)'''
    def reshape(H):
        indices = [0,1,2,4,5,7]
        eye = np.eye(3).reshape(9,)
        eye[indices] = H
        return eye.reshape(3,3)
    
    H_reshaped = np.array([reshape(H) for H in H_matrices]) # (B,3,3)

    return H_reshaped 


def postprocess_predictions(seg_masks, embeddings, H_matrices = None, max_num_lanes = 5, min_pixels = 15, min_bin_freq = 1, bandwidth = None):

    seg_masks = tf.nn.softmax(seg_masks, axis = -1)
    seg_masks = np.argmax(seg_masks, axis = -1) # (B,H,W,1)

    instance_masks, total_time = create_instance_masks(embeddings, seg_masks, max_num_lanes, min_pixels, min_bin_freq, bandwidth) # (B,H,W,1)

    if H_matrices is None:
        return seg_masks, instance_masks, total_time
    else:
        H_matrices = reshape_H(H_matrices) # (B,3,3)
        return seg_masks, instance_masks, H_matrices, total_time
