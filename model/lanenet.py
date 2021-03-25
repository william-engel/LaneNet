import os
import numpy as np

from enet import ENet_Encoder, ENet_Decoder
from hnet import HNet

# setup tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
tf.get_logger().setLevel('INFO')


def LaneNet(input_shape, num_classes, embedding_dim, train_seg = True, train_instance = True, train_hnet = False, include_hnet = True):
    input = tf.keras.Input(shape = input_shape)

    hnet_input_shape = (64,128)

    ### initialize multi-branch enet
    encoder          = ENet_Encoder(input_tensor = input)
    # semantic segmentation
    seg_decoder      = ENet_Decoder(input_tensor = encoder.output, classifier_activation = None, num_classes = num_classes, name = 'binary_decoder')
    seg_decoder.trainable = train_seg
    # instance segmentation
    instance_decoder = ENet_Decoder(input_tensor = encoder.output, classifier_activation = None, num_classes = embedding_dim, name = 'instance_decoder')
    instance_decoder.trainable = train_instance

    ### initialize hnet
    hnet = HNet(input_shape = input_shape, target_shape = hnet_input_shape)
    hnet.trainable = False

    ### build lanenet
    x = encoder(input)
    seg_output = seg_decoder(x)
    instance_output = instance_decoder(x)
    if include_hnet:
        h = hnet(input)
        outputs = [seg_output, instance_output, h]
    else:
        outputs = [seg_output, instance_output]
    model = tf.keras.models.Model(inputs = [input], outputs = outputs)

    # rename model outputs
    model.output_names[0] = 'binary_mask'
    model.output_names[1] = 'instance_mask'
    if include_hnet: model.output_names[2] = 'homography'

    return model


