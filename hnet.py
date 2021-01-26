import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
tf.get_logger().setLevel('INFO')

from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, MaxPool2D, Flatten, Dense
from tensorflow.keras.layers.experimental.preprocessing import Resizing 
from tensorflow.keras.initializers import Zeros  
from tensorflow.keras import Model


class CustomIdentity(tf.keras.initializers.Initializer):
  def __init__(self):
    super(CustomIdentity,self).__init__()
  
  def __call__(self, shape, dtype=None, **kwargs):
    return tf.constant([1, 0, 0, 1, 0 , 0], dtype = dtype)

  def get_config(self):
    return super(CustomIdentity, self).get_config()

def conv_block(input, filters, kernel_size, name):
  x = Conv2D(filters = filters, kernel_size = kernel_size, use_bias = False, name = name + '_conv')(input)
  x = BatchNormalization( name = name + '_bn')(x)
  x = ReLU(name = name + '_relu')(x)
  return x

def create_hnet(input_shape, target_shape = (64, 128)):
  TARGET_HEIGHT, TARGET_WIDTH =  target_shape
  
  input = Input(shape = input_shape)

  x = Resizing(TARGET_HEIGHT, TARGET_WIDTH, name = 'resize_input')(input)

  x = conv_block(input = x, filters = 16, kernel_size = 3, name = 'block1')
  x = conv_block(input = x, filters = 16, kernel_size = 3, name = 'block2')
  x = MaxPool2D(pool_size = (2,2), strides = (2,2), name = 'maxpool1')(x)

  x = conv_block(input = x, filters = 32, kernel_size = 3, name = 'block3')
  x = conv_block(input = x, filters = 32, kernel_size = 3, name = 'block4')
  x = MaxPool2D(pool_size = (2,2), strides = (2,2), name = 'maxpool2')(x)

  x = conv_block(input = x, filters = 64, kernel_size = 3, name = 'block5')
  x = conv_block(input = x, filters = 64, kernel_size = 3, name = 'block6')
  x = MaxPool2D(pool_size = (2,2), strides = (2,2), name = 'maxpool3')(x)

  x = Flatten(name = 'final_flatten')(x)
  x = Dense(units = 1024, name = 'final_dense')(x)
  x = BatchNormalization(name = 'final_bn')(x)
  x = ReLU(name = 'final_relu')(x)

  output = Dense(units = 6, name = 'output_dense', bias_initializer = CustomIdentity(), kernel_initializer = Zeros())(x)

  model = Model(inputs = [input], outputs = [output])
  return model
