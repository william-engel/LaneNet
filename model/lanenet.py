import os
import numpy as np

from enet import ENet_Encoder, ENet_Decoder
from hnet import HNet

# setup tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
tf.get_logger().setLevel('INFO')


def LaneNet(input_shape, num_classes, embedding_dim, train_seg = True, train_instance = True, train_hnet = False):
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
    h = hnet(input)
    seg_output = seg_decoder(x)
    instance_output = instance_decoder(x)
    model = tf.keras.models.Model(inputs = [input], outputs = [seg_output, instance_output, h])

    # rename model outputs
    model.output_names[0] = 'binary_mask'
    model.output_names[1] = 'instance_mask'
    model.output_names[2] = 'homography'

    return model




def  discriminative_loss():
    def loss(correct_label, prediction):

        def cond(label, batch, out_loss, out_var, out_dist, out_reg, i):
            return tf.less(i, tf.shape(batch)[0])

        def body(label, batch, out_loss, out_var, out_dist, out_reg, i):
            disc_loss, l_var, l_dist, l_reg = discriminative_loss_single(prediction[i], correct_label[i], feature_dim, image_shape, 
                            delta_v, delta_d, param_var, param_dist, param_reg)

            out_loss = out_loss.write(i, disc_loss)
            out_var = out_var.write(i, l_var)
            out_dist = out_dist.write(i, l_dist)
            out_reg = out_reg.write(i, l_reg)

            return label, batch, out_loss, out_var, out_dist, out_reg, i + 1

        # TensorArray is a data structure that support dynamic writing
        output_ta_loss = tf.TensorArray(dtype=tf.float32,
                        size=0,
                        dynamic_size=True)
        output_ta_var = tf.TensorArray(dtype=tf.float32,
                        size=0,
                        dynamic_size=True)
        output_ta_dist = tf.TensorArray(dtype=tf.float32,
                        size=0,
                        dynamic_size=True)
        output_ta_reg = tf.TensorArray(dtype=tf.float32,
                        size=0,
                        dynamic_size=True)

        _, _, out_loss_op, out_var_op, out_dist_op, out_reg_op, _  = tf.while_loop(cond, body, [correct_label, 
                                                            prediction, 
                                                            output_ta_loss, 
                                                            output_ta_var, 
                                                            output_ta_dist, 
                                                            output_ta_reg, 
                                                            0])
        out_loss_op = out_loss_op.stack()
        out_var_op = out_var_op.stack()
        out_dist_op = out_dist_op.stack()
        out_reg_op = out_reg_op.stack()
        
        disc_loss = tf.reduce_mean(out_loss_op)
        l_var = tf.reduce_mean(out_var_op)
        l_dist = tf.reduce_mean(out_dist_op)
        l_reg = tf.reduce_mean(out_reg_op)

        return disc_loss
    return loss

