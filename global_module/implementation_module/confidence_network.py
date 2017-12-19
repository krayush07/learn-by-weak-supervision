import tensorflow as tf

from global_module.settings_module import ParamsClass, Directory


class ConfidenceNetwork:
    def compute_confidence(self, feature_input, num_layers):
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
            final_logits = tf.contrib.layers.fully_connected(fc_output, 1)
            predicted_score = tf.sigmoid(final_logits, name='cnf_score')
            return final_logits, predicted_score


    def compute_loss(self, true_label, weak_label, logits, num_classes):
        one_hot_true_label = tf.one_hot(true_label, num_classes)
        target_score = tf.subtract(tf.constant(1.), tf.reduce_mean(tf.abs(tf.subtract(one_hot_true_label, weak_label)), axis=1))
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=target_score, logits=logits, name='confidence_loss')
        return loss