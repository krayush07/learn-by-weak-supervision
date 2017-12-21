import tensorflow as tf

from global_module.settings_module import ParamsClass, Directory


class TargetNetwork:
    def __init__(self, optimizer, max_grad_norm):
        self.optimizer = optimizer
        self.max_grad_norm = max_grad_norm

    def predict_labels(self, feature_input, num_layers, num_classes):
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


    def compute_loss(self, logits, labels, confidence):
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='target_loss')
        weighted_loss = tf.multiply(confidence, loss, name='weighted_loss')
        return weighted_loss

    def train(self, feature_input, num_layers, num_classes, confidence, weak_label, lr):
        global optimizer
        final_logits = self.predict_labels(feature_input, num_layers, num_classes)
        loss = self.compute_loss(final_logits, weak_label, confidence)

        with tf.variable_scope('optimize_cnf_net'):
            tvars = tf.trainable_variables()
            # TODO: Check if target network params are const.
            # TODO: Add regularization loss.
            grads = tf.gradients(loss, tvars)
            grads, _ = tf.clip_by_global_norm(grads, clip_norm=self.max_grad_norm)
            grad_var_pairs = zip(grads, tvars)

            if self.optimizer == 'sgd':
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr, name='sgd')
            elif self.optimizer == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate=lr, name='adam')
            elif self.optimizer == 'adadelta':
                optimizer = tf.train.AdadeltaOptimizer(learning_rate=lr, epsilon=1e-6, name='adadelta')
            return optimizer.apply_gradients(grad_var_pairs, name='apply_grad')