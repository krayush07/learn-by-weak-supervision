import tensorflow as tf

from global_module.settings_module import GlobalParams, Directory


class ConfidenceNetwork:
    def __init__(self, num_classes, optimizer, max_grad_norm=5, reg_const=0.001, log=False):
        self.num_classes = num_classes
        self.optimizer = optimizer
        self.max_grad_norm = max_grad_norm
        self.reg_const = reg_const
        self._lr = tf.Variable(0.01, trainable=False, name='learning_rate')
        self.log = log

    def compute_confidence(self, feature_input, num_layers):
        with tf.variable_scope('fc_layer1'):
            fc_output = tf.layers.dense(feature_input, 32, activation=tf.nn.leaky_relu, bias_initializer=tf.constant_initializer(0.1))

        if num_layers >= 2:
            with tf.variable_scope('fc_layer2'):
                fc_output = tf.layers.dense(fc_output, 32, activation=tf.nn.leaky_relu, bias_initializer=tf.constant_initializer(0.1))

        if num_layers >= 3:
            with tf.variable_scope('fc_layer3'):
                fc_output = tf.layers.dense(fc_output, 32, activation=tf.nn.leaky_relu, bias_initializer=tf.constant_initializer(0.1))

        if num_layers >= 4:
            with tf.variable_scope('fc_layer4'):
                fc_output = tf.layers.dense(fc_output, 32, activation=tf.nn.leaky_relu, bias_initializer=tf.constant_initializer(0.01))

        if num_layers >= 5:
            with tf.variable_scope('fc_layer5'):
                fc_output = tf.layers.dense(fc_output, 32, activation=tf.nn.leaky_relu, bias_initializer=tf.constant_initializer(0.01))

        with tf.variable_scope('fc_layer_final'):
            final_logits = tf.layers.dense(fc_output, 1, activation=tf.nn.leaky_relu, bias_initializer=tf.constant_initializer(0.01))
            predicted_score = tf.sigmoid(final_logits, name='cnf_score')

        return final_logits, predicted_score, fc_output

    def compute_xent_loss(self, true_label, weak_label, logits):
        one_hot_true_label = tf.one_hot(true_label, self.num_classes)
        target_score = tf.subtract(tf.constant(1.), tf.reduce_mean(tf.abs(tf.subtract(one_hot_true_label, weak_label)), axis=1))
        self.indiv_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.squeeze(target_score), logits=tf.squeeze(logits), name='confidence_loss')
        self.tg = tf.squeeze(target_score)
        self.lg = tf.squeeze(logits)
        self.sb = tf.subtract(one_hot_true_label, weak_label)
        self.one_hot = one_hot_true_label
        return self.indiv_loss

    def train(self, feature_input, num_layers, true_label, weak_label):
        global optimizer
        trainable_tvars, total_loss = self.aggregate_loss(feature_input, num_layers, true_label, weak_label)

        with tf.variable_scope('optimize_cnf_net'):
            grads = tf.gradients(total_loss, trainable_tvars)
            grads, _ = tf.clip_by_global_norm(grads, clip_norm=self.max_grad_norm)
            grad_var_pairs = zip(grads, trainable_tvars)

            if self.optimizer == 'sgd':
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr, name='sgd')
            elif self.optimizer == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, name='adam')
            elif self.optimizer == 'adadelta':
                optimizer = tf.train.AdadeltaOptimizer(learning_rate=self.lr, epsilon=1e-6, name='adadelta')
            train_op = optimizer.apply_gradients(grad_var_pairs, name='apply_grad')

            if self.log:
                self.train_loss = tf.summary.scalar('loss_train', total_loss)

            if self.log:
                grad_summaries = []
                for grad, var in grad_var_pairs:
                    if grad is not None:
                        grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(var.name), grad)
                        sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(var.name), tf.nn.zero_fraction(grad))
                        grad_summaries.append(grad_hist_summary)
                        grad_summaries.append(sparsity_summary)
                grad_summaries_merged = tf.summary.merge(grad_summaries)

                self.merged_train = tf.summary.merge([self.train_loss, grad_summaries_merged])
            else:
                self.merged_train = []

            return train_op
            # return optimizer.apply_gradients(grad_var_pairs, name='apply_grad')

    def aggregate_loss(self, feature_input, num_layers, true_label, weak_label):
        self.feat_input = feature_input
        final_logits, confidence_score, fc_op = self.compute_confidence(feature_input, num_layers)
        self.fc_op = fc_op
        self.final_lg = final_logits
        self.cnf_sc = confidence_score
        loss = tf.reduce_mean(self.compute_xent_loss(true_label, weak_label, final_logits))

        global_tvars = tf.trainable_variables()

        # (Done) TODO: Check if target network params are const.
        tar_net_tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "classifier/tar_net")
        trainable_tvars = list(set(global_tvars) - set(tar_net_tvars))

        # (Done) TODO: Add regularization loss.
        l2_regularizer = tf.contrib.layers.l2_regularizer(scale=self.reg_const, scope="cnf_reg")
        regularization_penalty = tf.contrib.layers.apply_regularization(l2_regularizer, trainable_tvars)
        # total_loss = loss + regularization_penalty
        self.total_loss = loss + regularization_penalty
        return trainable_tvars, self.total_loss

    def assign_lr(self, session, lr_value):
        session.run(tf.assign(self.lr, lr_value))

    @property
    def lr(self):
        return self._lr
