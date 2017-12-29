import tensorflow as tf

from global_module.settings_module import GlobalParams, Directory
from global_module.implementation_module import CNN


class RepLayer:
    def __init__(self):
        self.cnn = CNN()

    def create_representation(self, conv_input, params):
        conv_input = tf.expand_dims(conv_input, -1)
        filter_width = params.filter_width
        conv_stride = [1, 1, 1, 1]
        num_filters = params.num_filters
        conv_padding = params.conv_padding
        ksize = [1, params.pool_width, 1, 1]
        pool_stride = [1, params.pool_stride, 1, 1]
        pool_padding = params.pool_padding

        with tf.variable_scope('create_rep'):
            pool_output = []
            for i in range(len(filter_width)):
                kernel = [filter_width[i], 300, 1, num_filters]
                curr_convolution_output = self.cnn.conv_output(conv_input, kernel, conv_stride, num_filters, conv_padding, 'conv0_filter' + str(i))
                curr_pool_output = self.cnn.pool_output(curr_convolution_output, ksize, pool_stride, pool_padding, 'pool0_filter' + str(i))
                pool_output.append(curr_pool_output)
            concatenated_pool_output = tf.concat(pool_output, axis=1)
            return tf.reshape(concatenated_pool_output, shape=[-1, concatenated_pool_output.shape.dims[1].value * concatenated_pool_output.shape.dims[3].value])
