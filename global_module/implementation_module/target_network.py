import tensorflow as tf

from global_module.settings_module import GlobalParams, Directory


class TargetNetwork:
    def __init__(self, optimizer, max_grad_norm=5, reg_const=0.001):
        self.optimizer = optimizer
        self.max_grad_norm = max_grad_norm
        self.reg_const = reg_const
        self._lr = tf.Variable(0.01, trainable=False, name='learning_rate')

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
        self.indiv_loss = tf.multiply(confidence, loss, name='weighted_loss')
        return self.indiv_loss

    def train(self, feature_input, num_layers, num_classes, confidence, weak_label):
        global optimizer
        trainable_tvars, total_loss = self.aggregate_loss(feature_input, num_layers, num_classes, confidence, weak_label)
        with tf.variable_scope('optimize_tar_net'):

            grads = tf.gradients(total_loss, trainable_tvars)
            grads, _ = tf.clip_by_global_norm(grads, clip_norm=self.max_grad_norm)
            grad_var_pairs = zip(grads, trainable_tvars)

            if self.optimizer == 'sgd':
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr, name='sgd')
            elif self.optimizer == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, name='adam')
            elif self.optimizer == 'adadelta':
                optimizer = tf.train.AdadeltaOptimizer(learning_rate=self.lr, epsilon=1e-6, name='adadelta')
            return optimizer.apply_gradients(grad_var_pairs, name='apply_grad')

    def aggregate_loss(self, feature_input, num_layers, num_classes, confidence, weak_label):
        final_logits = self.predict_labels(feature_input, num_layers, num_classes)
        loss = tf.reduce_mean(self.compute_loss(final_logits, weak_label, confidence))

        global_tvars = tf.trainable_variables()

        # (Done) TODO: Check if target network params are const.
        cnf_net_tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "cnf_net")
        trainable_tvars = list(set(global_tvars) - set(cnf_net_tvars))

        # (Done) TODO: Add regularization loss.
        l2_regularizer = tf.contrib.layers.l2_regularizer(scale=self.reg_const, scope="tar_reg")
        regularization_penalty = tf.contrib.layers.apply_regularization(l2_regularizer, trainable_tvars)
        self.total_loss = loss + regularization_penalty
        return trainable_tvars, self.total_loss

    def assign_lr(self, session, lr_value):
        session.run(tf.assign(self.lr, lr_value))

    @property
    def lr(self):
        return self._lr

        # @property
        # def train_op(self):
        #     return self._train_op
