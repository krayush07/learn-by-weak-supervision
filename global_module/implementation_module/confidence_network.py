import tensorflow as tf

from global_module.settings_module import ParamsClass, Directory


class ConfidenceNetwork:
    def __init__(self, num_classes, optimizer=None, max_grad_norm=5):
        self.num_classes = num_classes
        self.optimizer = optimizer
        self.max_grad_norm = max_grad_norm

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

    def compute_loss(self, true_label, weak_label, logits):
        one_hot_true_label = tf.one_hot(true_label, self.num_classes)
        target_score = tf.subtract(tf.constant(1.), tf.reduce_mean(tf.abs(tf.subtract(one_hot_true_label, weak_label)), axis=1))
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.squeeze(target_score), logits=logits, name='confidence_loss')
        return loss

    def train(self, feature_input, num_layers, true_label, weak_label, lr):
        global optimizer
        final_logits, confidence_score = self.compute_confidence(feature_input, num_layers)
        loss = self.compute_loss(true_label, weak_label, final_logits)

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
