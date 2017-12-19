import tensorflow as tf

from global_module.settings_module import ParamsClass, Directory


class TargetNetwork:
    def compute_confidence(self, feature_input, num_layers, num_classes):
        with tf.variable_scope('fc_layer1'):
            fc_output = tf.contrib.layers.fully_connected(feature_input, 512)

        if num_layers >= 2:
            with tf.variable_scope('fc_layer2'):
                fc_output = tf.contrib.layers.fully_connected(fc_output, 256)

        if num_layers >= 3:
            with tf.variable_scope('fc_layer3'):
                fc_output = tf.contrib.layers.fully_connected(fc_output, 256)

        if num_layers >= 4:
            with tf.variable_scope('fc_layer4'):
                fc_output = tf.contrib.layers.fully_connected(fc_output, 128)

        if num_layers >= 5:
            with tf.variable_scope('fc_layer5'):
                fc_output = tf.contrib.layers.fully_connected(fc_output, 64)

        with tf.variable_scope('fc_layer_final'):
            final_logits = tf.contrib.layers.fully_connected(fc_output, num_classes)
            return final_logits


    def compute_loss(self, logits, labels, confidence, num_classes):
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='target_loss')
        weighted_loss = tf.multiply(confidence, loss, name='weighted_loss')
        return weighted_loss