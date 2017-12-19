import tensorflow as tf

from global_module.settings_module import ParamsClass, Directory
from global_module.implementation_module import CNN

class RepLayer:
    def __init__(self):
        self.cnn = CNN()

    def create_representation(self, conv_input, kernel, conv_stride, num_filters, conv_padding, ksize, pool_stride, pool_padding):
        with tf.variable_scope('create_rep'):
            convolution_output = self.cnn.conv_output(conv_input, kernel, conv_stride, num_filters, conv_padding, 'conv')
            pool_output = self.cnn.pool_output(convolution_output, ksize, pool_stride, pool_padding, 'pool')
            return pool_output