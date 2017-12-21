import tensorflow as tf

class CNN:
    def conv_output(self, conv_input, kernel, strides, num_filters, padding, name):
        return self._conv_layer(conv_input, kernel, num_filters, strides, padding, name)

    def pool_output(self, pool_input, ksize, stride, padding, name):
        return tf.nn.max_pool(pool_input, ksize, stride, padding, name=name)

    def _conv_layer(self, conv_input, filter_shape, num_filters, stride, padding, name):
        with tf.variable_scope(name) as scope:
            try:
                weights = tf.get_variable(name='weights', shape=filter_shape, regularizer=tf.contrib.layers.l2_regularizer(scale=0.01),
                                          initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
                biases = tf.get_variable(name='biases', shape=[num_filters], regularizer=tf.contrib.layers.l2_regularizer(0.0),
                                         initializer=tf.constant_initializer(0.0))
            except ValueError:
                scope.reuse_variables()
                weights = tf.get_variable(name='weights', shape=filter_shape, regularizer=tf.contrib.layers.l2_regularizer(scale=0.01),
                                          initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
                biases = tf.get_variable(name='biases', shape=[num_filters], regularizer=tf.contrib.layers.l2_regularizer(0.0),
                                         initializer=tf.constant_initializer(0.0))
            # conv_input_nhwc = tf.transpose(conv_input, perm=[1,2,3,0], name='conv_input_nhwc')
            # kernel = tf.transpose(weights, perm=[0,2,1,3], name='kernel_nhwc')
            # conv = tf.nn.conv2d(conv_input, filter=weights, strides=stride, padding=padding)
            conv = tf.nn.conv2d(conv_input, filter=weights, strides=stride, padding=padding)
            # scope.reuse_variables()
            return tf.nn.relu(conv + biases)