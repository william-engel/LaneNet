import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import MaxPool2D, Conv2D, Input, BatchNormalization, Activation, PReLU, Add, SpatialDropout2D, Conv2DTranspose, Activation, ReLU, UpSampling2D
from tensorflow.nn import max_pool_with_argmax
from tensorflow.keras.models import Model
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.ops import nn_ops
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.keras import backend as K
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras.losses import categorical_crossentropy
from tf.keras.regularizers import L2

def ENet_Encoder(input_shape = None, input_tensor = None, name = 'encoder'):

    ## INPUT
    if input_tensor is None:
      input = Input(shape = input_shape)
    else:
      input = input_tensor

    ## INITIAL
    x = initial(input, name = "initial")

    ## STAGE 1
    x, idx_1 = bottleneck(input = x, filters = 64, kernel_size = 3, dropout_rate = 0.01, downsampling=True, name = "bn1.0")
    x = bottleneck(input = x, filters = 64, kernel_size = 3, dropout_rate = 0.01, name = "bn1.1")
    x = bottleneck(input = x, filters = 64, kernel_size = 3, dropout_rate = 0.01, name = "bn1.2")
    x = bottleneck(input = x, filters = 64, kernel_size = 3, dropout_rate = 0.01, name = "bn1.3")
    x = bottleneck(input = x, filters = 64, kernel_size = 3, dropout_rate = 0.01, name = "bn1.4")

    ## STAGE 2
    x, idx_2 = bottleneck(input = x, filters = 128, kernel_size = 3, dropout_rate = 0.1, downsampling=True, name = "bn2.0")
    x = bottleneck(input = x, filters = 128, kernel_size = 3, dropout_rate = 0.1, name = "bn2.1")  
    x = bottleneck(input = x, filters = 128, kernel_size = 3, dropout_rate = 0.1, dilation_rate = 2, dilated = True, name = "bn2.2")
    x = bottleneck(input = x, filters = 128, kernel_size = 5, dropout_rate = 0.1, asymmetric = True, name = "bn2.3")
    x = bottleneck(input = x, filters = 128, kernel_size = 3, dropout_rate = 0.1, dilation_rate = 4, dilated = True, name = "bn2.4")
    x = bottleneck(input = x, filters = 128, kernel_size = 3, dropout_rate = 0.1, name = "bn2.5")
    x = bottleneck(input = x, filters = 128, kernel_size = 3, dropout_rate = 0.1, dilation_rate = 8, dilated = True, name = "bn2.6")
    x = bottleneck(input = x, filters = 128, kernel_size = 5, dropout_rate = 0.1, asymmetric = True, name = "bn2.7")
    output = bottleneck(input = x, filters = 128, kernel_size = 3, dropout_rate = 0.1, dilation_rate = 16, dilated = True, name = "bn2.8")

    model = Model(inputs = [input], outputs = [output, idx_2, idx_1], name = name)

    return model

def ENet_Decoder(input_shape = None, input_tensor = None, include_top=True, classifier_activation='softmax', num_classes=10, name = 'decoder'):

    ## INPUT
    if input_tensor is None:
      inputs = [Input(shape = shape) for shape in input_shape]
    else:
      inputs = [Input(tensor = tensor) for tensor in input_tensor]


    ## STAGE 3
    x = bottleneck(input = inputs[0], filters = 128, kernel_size = 3, dropout_rate = 0.1, name = "bn3.0")  
    x = bottleneck(input = x, filters = 128, kernel_size = 3, dropout_rate = 0.1, dilation_rate = 2, dilated = True, name = "bn3.1")
    x = bottleneck(input = x, filters = 128, kernel_size = 5, dropout_rate = 0.1, asymmetric = True, name = "bn3.2")
    x = bottleneck(input = x, filters = 128, kernel_size = 3, dropout_rate = 0.1, dilation_rate = 4, dilated = True, name = "bn3.3")
    x = bottleneck(input = x, filters = 128, kernel_size = 3, dropout_rate = 0.1, name = "bn3.4")
    x = bottleneck(input = x, filters = 128, kernel_size = 3, dropout_rate = 0.1, dilation_rate = 8, dilated = True, name = "bn.5")
    x = bottleneck(input = x, filters = 128, kernel_size = 5, dropout_rate = 0.1, asymmetric = True, name = "bn3.6")
    x = bottleneck(input = x, filters = 128, kernel_size = 3, dropout_rate = 0.1, dilation_rate = 16, dilated = True, name = "bn3.7")

    ## STAGE 4
    x = bottleneck(input = x, filters = 64, kernel_size = 3, dropout_rate = 0.1, idx = inputs[1], upsampling = True, name = "bn4.0")
    x = bottleneck(input = x, filters = 64, kernel_size = 3, dropout_rate = 0.1, name = "bn4.1")
    x = bottleneck(input = x, filters = 64, kernel_size = 3, dropout_rate = 0.1, name = "bn4.2") 

    ## STAGE 5
    x = bottleneck(input = x, filters = 16, kernel_size = 3, dropout_rate = 0.1, idx = inputs[2], upsampling = True, name = "bn5.0")
    x = bottleneck(input = x, filters = 16, kernel_size = 3, dropout_rate = 0.1, name = "bn5.1")  

    ## OUTPUT
    if include_top:
        x = Conv2DTranspose(filters = num_classes, kernel_size = (2,2), strides = (2,2), padding = 'same')(x)
        output = Activation(classifier_activation)(x)
    else:
        output = x

    model = Model(inputs = inputs, outputs = output, name = name)
  
    return model  

def ENet(input_shape=None, include_top=True, classifier_activation='softmax', num_classes=10):

    ## INPUT
    input = Input(shape = input_shape, name = "initial_input")
    x = initial(input, name = "initial")

    ## STAGE 1
    x, idx_1 = bottleneck(input = x, filters = 64, kernel_size = 3, dropout_rate = 0.01, downsampling=True, name = "bn1.0")
    x = bottleneck(input = x, filters = 64, kernel_size = 3, dropout_rate = 0.01, name = "bn1.1")
    x = bottleneck(input = x, filters = 64, kernel_size = 3, dropout_rate = 0.01, name = "bn1.2")
    x = bottleneck(input = x, filters = 64, kernel_size = 3, dropout_rate = 0.01, name = "bn1.3")
    x = bottleneck(input = x, filters = 64, kernel_size = 3, dropout_rate = 0.01, name = "bn1.4")

    ## STAGE 2
    x, idx_2 = bottleneck(input = x, filters = 128, kernel_size = 3, dropout_rate = 0.1, downsampling=True, name = "bn2.0")
    x = bottleneck(input = x, filters = 128, kernel_size = 3, dropout_rate = 0.1, name = "bn2.1")  
    x = bottleneck(input = x, filters = 128, kernel_size = 3, dropout_rate = 0.1, dilation_rate = 2, dilated = True, name = "bn2.2")
    x = bottleneck(input = x, filters = 128, kernel_size = 5, dropout_rate = 0.1, asymmetric = True, name = "bn2.3")
    x = bottleneck(input = x, filters = 128, kernel_size = 3, dropout_rate = 0.1, dilation_rate = 4, dilated = True, name = "bn2.4")
    x = bottleneck(input = x, filters = 128, kernel_size = 3, dropout_rate = 0.1, name = "bn2.5")
    x = bottleneck(input = x, filters = 128, kernel_size = 3, dropout_rate = 0.1, dilation_rate = 8, dilated = True, name = "bn2.6")
    x = bottleneck(input = x, filters = 128, kernel_size = 5, dropout_rate = 0.1, asymmetric = True, name = "bn2.7")
    x = bottleneck(input = x, filters = 128, kernel_size = 3, dropout_rate = 0.1, dilation_rate = 16, dilated = True, name = "bn2.8")

    ## STAGE 3
    x = bottleneck(input = x, filters = 128, kernel_size = 3, dropout_rate = 0.1, name = "bn3.0")  
    x = bottleneck(input = x, filters = 128, kernel_size = 3, dropout_rate = 0.1, dilation_rate = 2, dilated = True, name = "bn3.1")
    x = bottleneck(input = x, filters = 128, kernel_size = 5, dropout_rate = 0.1, asymmetric = True, name = "bn3.2")
    x = bottleneck(input = x, filters = 128, kernel_size = 3, dropout_rate = 0.1, dilation_rate = 4, dilated = True, name = "bn3.3")
    x = bottleneck(input = x, filters = 128, kernel_size = 3, dropout_rate = 0.1, name = "bn3.4")
    x = bottleneck(input = x, filters = 128, kernel_size = 3, dropout_rate = 0.1, dilation_rate = 8, dilated = True, name = "bn.5")
    x = bottleneck(input = x, filters = 128, kernel_size = 5, dropout_rate = 0.1, asymmetric = True, name = "bn3.6")
    x = bottleneck(input = x, filters = 128, kernel_size = 3, dropout_rate = 0.1, dilation_rate = 16, dilated = True, name = "bn3.7")

    ## STAGE 4
    x = bottleneck(input = x, filters = 64, kernel_size = 3, dropout_rate = 0.1, idx = idx_2, upsampling = True, name = "bn4.0")
    x = bottleneck(input = x, filters = 64, kernel_size = 3, dropout_rate = 0.1, name = "bn4.1")
    x = bottleneck(input = x, filters = 64, kernel_size = 3, dropout_rate = 0.1, name = "bn4.2") 

    ## STAGE 5
    x = bottleneck(input = x, filters = 16, kernel_size = 3, dropout_rate = 0.1, idx = idx_1, upsampling = True, name = "bn5.0")
    x = bottleneck(input = x, filters = 16, kernel_size = 3, dropout_rate = 0.1, name = "bn5.1")  

  ## OUTPUT
    if include_top:
        x = Conv2DTranspose(filters = num_classes, kernel_size = (2,2), strides = (2,2), padding = 'same')(x)
        output = Activation(classifier_activation)(x)
    else:
        output = x

    model = Model(inputs = [input], outputs = [output])
  
    return model


def initial(input, name):
  x_conv = Conv2D(filters = 13, kernel_size = (3,3), strides = (2,2), padding = 'same', kernel_regularizer = L2(0.001), name = name + '_3x3conv')(input)
  x_conv = BatchNormalization(name = name + '_bnorm', fused = True)(x_conv)
  x_conv = PReLU(name = name + '_prelu')(x_conv)

  x_pool = MaxPool2D(pool_size = (2,2), strides = 2, name = name + '_maxp')(input)

  x = tf.concat([x_conv, x_pool], axis = -1) 
  return x

def bottleneck(input, filters, kernel_size, dropout_rate, name, dilation_rate = 1, idx = [],  
               downsampling = False, upsampling = False, dilated = False, asymmetric = False):

  # decrease dimensionality
  reduce_factor = 4
  filters_reduced = filters / reduce_factor

  ### DOWNSAMPLING ###
  if downsampling == True:
    name = name + '_down'
            
    # MAIN BRANCH
    x_main, idx = max_pool_with_argmax(input, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME', name = name + '_main_maxp')

    pad_depth = abs(filters - input.shape[-1])
    paddings = [[0,0], [0,0], [0,0], [0,pad_depth]]
    x_main = tf.pad(x_main, paddings)

    # SUB BRANCH
    x = Conv2D(filters = filters_reduced, kernel_size = (2,2), strides = (2,2), padding = 'same', kernel_regularizer = L2(0.001), name = name + '_2x2conv')(input)
    x = BatchNormalization(name = name + '_bnorm_1')(x)
    x = PReLU(name = name + '_prelu_1')(x)

    x = Conv2D(filters = filters_reduced, kernel_size = kernel_size, strides = (1,1), padding = 'same', kernel_regularizer = L2(0.001), name = name + '_3x3conv')(x)
    x = BatchNormalization(name = name + '_bnorm_2')(x)
    x = PReLU(name = name + '_prelu_2')(x)

    x = Conv2D(filters = filters, kernel_size = (1,1), strides = (1,1), padding = 'same', kernel_regularizer = L2(0.001), name = name + '_1x1conv')(x)
    x = BatchNormalization(name = name + '_bnorm_3')(x)
    x = PReLU(name = name + '_prelu_3')(x)

    x = SpatialDropout2D(dropout_rate, name = name + '_drop')(x)

    x = Add(name = name + '_add')([x, x_main])
    x = PReLU(name = name + '_prelu_4')(x)

    return x, idx

  ### UPSAMPLING ###
  if upsampling == True:
    name = name + '_up'

    # MAIN BRANCH
    x_main = Conv2D(filters = filters, kernel_size = (1,1), padding = 'same', kernel_regularizer = L2(0.001),  name = name + '_main_1x1conv')(input)
    x_main = BatchNormalization(name = name + '_main_bnorm')(x_main)
    x_main = MaxUnpooling2D()([x_main, idx])

    # SUB BRANCH
    x = Conv2D(filters = filters_reduced, kernel_size = (1,1), strides = (1,1), padding = 'same', kernel_regularizer = L2(0.001), name = name + '_1x1downconv')(input)
    x = BatchNormalization(name = name + '_bnorm_1')(x)
    x = PReLU(name = name + '_prelu_1')(x)

    x = Conv2DTranspose(filters = filters_reduced, kernel_size = kernel_size, strides = (2,2), padding = 'same', kernel_regularizer = L2(0.001), name = name + '_3x3transconv')(input)
    x = BatchNormalization(name = name + '_bnorm_1')(x)
    x = ReLU(name = name + '_relu_1')(x)

    x = Conv2D(filters = filters, kernel_size = (1,1), strides = (1,1), padding = 'same', kernel_regularizer = L2(0.001), name = name + '_1x1conv')(x)
    x = BatchNormalization(name = name + '_bnorm_2')(x)
    x = ReLU(name = name + '_relu_2')(x)

    x = SpatialDropout2D(dropout_rate, name = name + '_drop')(x)
    x = ReLU(name = name + '_relu_3')(x)

    x = Add(name = name + '_add')([x, x_main])
    x = ReLU(name = name + '_relu_4')(x)

    return x

  ### DILATED ###
  elif dilated == True:
    name = name + '_dilated'

    x = Conv2D(filters = filters_reduced, kernel_size = (1,1), strides = (1,1), padding = 'same', kernel_regularizer = L2(0.001), name = name + '_1x1downconv')(input)
    x = BatchNormalization(name = name + '_bnorm_1')(x)
    x = PReLU(name = name + '_prelu_1')(x)

    x = Conv2D(filters = filters_reduced, kernel_size = kernel_size, strides = (1,1), padding = 'same', kernel_regularizer = L2(0.001), dilation_rate = dilation_rate, name = name + '_3x3conv')(x)
    x = BatchNormalization(name = name + '_bnorm_2')(x)
    x = PReLU(name = name + '_prelu_2')(x)

    x = Conv2D(filters = filters, kernel_size = (1,1), strides = (1,1), padding = 'same', kernel_regularizer = L2(0.001), name = name + '_1x1upconv')(x)
    x = BatchNormalization(name = name + '_bnorm_3')(x)
    x = PReLU(name = name + '_prelu_3')(x)

    x = SpatialDropout2D(dropout_rate, name = name + '_drop')(x)
    x = PReLU(name = name + '_prelu_4')(x)

    x = Add(name = name + '_add')([input, x])
    x = PReLU(name = name + '_prelu_5')(x)

  ### ASYMMETRIC ###
  elif asymmetric == True:
    name = name + '_asym'

    x = Conv2D(filters = filters_reduced, kernel_size = (1,1), strides = (1,1), padding = 'same', kernel_regularizer = L2(0.001), name = name + '_1x1downconv')(input)
    x = BatchNormalization(name = name + '_bnorm_1')(x)
    x = PReLU(name = name + '_prelu_1')(x)

    x = Conv2D(filters = filters_reduced, kernel_size = (kernel_size, 1) , strides = (1,1), padding = 'same', kernel_regularizer = L2(0.001), name = name + '_5x1conv')(x)
    x = Conv2D(filters = filters_reduced, kernel_size = (1, kernel_size) , strides = (1,1), padding = 'same', kernel_regularizer = L2(0.001), name = name + '_1x5conv')(x)
    x = BatchNormalization(name = name + '_bnorm_2')(x)
    x = PReLU(name = name + '_prelu_2')(x)

    x = Conv2D(filters = filters, kernel_size = (1,1), strides = (1,1), padding = 'same', kernel_regularizer = L2(0.001), name = name + '_1x1upconv')(x)
    x = BatchNormalization(name = name + '_bnorm_3')(x)
    x = PReLU(name = name + '_prelu_3')(x)

    x = SpatialDropout2D(dropout_rate, name = name + '_drop')(x)
    x = PReLU(name = name + '_prelu_4')(x)

    x = Add(name = name + '_add')([input, x])
    x = PReLU(name = name + '_prelu_5')(x)

  ### REGULAR ###
  else:
    name = name + '_regular'

    x = Conv2D(filters = filters_reduced, kernel_size = (1,1), strides = (1,1), padding = 'same', kernel_regularizer = L2(0.001), name = name + '_1x1downconv')(input)
    x = BatchNormalization(name = name + '_bnorm_1')(x)
    x = PReLU(name = name + '_prelu_1')(x)

    x = Conv2D(filters = filters_reduced, kernel_size = kernel_size, strides = (1,1), padding = 'same', kernel_regularizer = L2(0.001), name = name + '_3x3conv')(x)
    x = BatchNormalization(name = name + '_bnorm_2')(x)
    x = PReLU(name = name + '_prelu_2')(x)

    x = Conv2D(filters = filters, kernel_size = (1,1), strides = (1,1), padding = 'same', kernel_regularizer = L2(0.001), name = name + '_1x1upconv')(x)
    x = BatchNormalization(name = name + '_bnorm_3')(x)
    x = PReLU(name = name + '_prelu_3')(x)

    x = SpatialDropout2D(dropout_rate, name = name + '_drop')(x)
    x = PReLU(name = name + '_prelu_4')(x)

    x = Add(name = name + '_add')([input, x])
    x = PReLU(name = name + '_prelu_5')(x)

  return x

class MaxUnpooling2D(Layer):

    def __init__(self, size=(2, 2), **kwargs):
        super(MaxUnpooling2D, self).__init__(**kwargs)
        self.size = conv_utils.normalize_tuple(size, 2, 'size')

    def call(self, inputs, output_shape=None):
        updates, mask = inputs[0], inputs[1]
        
        mask = tf.cast(mask, 'int32')
        input_shape = tf.shape(updates, out_type='int32')
        #  calculation new shape
        if output_shape is None:
            output_shape = (input_shape[0], input_shape[1] * self.size[0], input_shape[2] * self.size[1], input_shape[3])
        
        # calculation indices for batch, height, width and feature maps
        one_like_mask = K.ones_like(mask, dtype='int32')
        batch_shape = K.concatenate([[input_shape[0]], [1], [1], [1]], axis=0)
        batch_range = K.reshape(tf.range(output_shape[0], dtype='int32'), shape=batch_shape)
        b = one_like_mask * batch_range
        y = mask // (output_shape[2] * output_shape[3])
        x = (mask // output_shape[3]) % output_shape[2]
        feature_range = tf.range(output_shape[3], dtype='int32')
        f = one_like_mask * feature_range
        
        # transpose indices & reshape update values to one dimension
        updates_size = tf.size(updates)
        indices = K.transpose(K.reshape(K.stack([b, y, x, f]), [4, updates_size]))
        values = K.reshape(updates, [updates_size])
        ret = tf.scatter_nd(indices, values, output_shape)
        return ret
    
    def get_config(self):
        config = super(MaxUnpooling2D, self).get_config()
        config.update({
            'size': self.size,
        })
        return config