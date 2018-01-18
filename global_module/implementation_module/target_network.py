import tensorflow as tf


class TargetNetwork:
    def __init__(self, optimizer, max_grad_norm=5, reg_const=0.001, log=False):
        self.optimizer = optimizer
        self.max_grad_norm = max_grad_norm
        self.reg_const = reg_const
        self._lr = tf.Variable(0.01, trainable=False, name='learning_rate')
        self.log = log

    def predict_labels(self, feature_input, num_layers, num_classes):
        with tf.variable_scope('fc_layer1'):
            fc_output = tf.layers.dense(feature_input, 64, activation=tf.nn.leaky_relu, bias_initializer=tf.constant_initializer(0.1))

        if num_layers >= 2:
            with tf.variable_scope('fc_layer2'):
                fc_output = tf.layers.dense(fc_output, 64, activation=tf.nn.leaky_relu, bias_initializer=tf.constant_initializer(0.1))

        if num_layers >= 3:
            with tf.variable_scope('fc_layer3'):
                fc_output = tf.layers.dense(fc_output, 64, activation=tf.nn.leaky_relu, bias_initializer=tf.constant_initializer(0.1))

        if num_layers >= 4:
            with tf.variable_scope('fc_layer4'):
                fc_output = tf.layers.dense(fc_output, 64, activation=tf.nn.leaky_relu, bias_initializer=tf.constant_initializer(0.1))

        if num_layers >= 5:
            with tf.variable_scope('fc_layer5'):
                fc_output = tf.layers.dense(fc_output, 64, activation=tf.nn.leaky_relu, bias_initializer=tf.constant_initializer(0.1))

        with tf.variable_scope('fc_layer_final'):
            final_logits = tf.layers.dense(fc_output, num_classes, activation=tf.nn.leaky_relu, bias_initializer=tf.constant_initializer(0.1))
            return final_logits

    def compute_loss(self, logits, labels, confidence):
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=tf.nn.softmax(labels), logits=logits, name='target_loss')
        # self.indiv_loss = tf.multiply(confidence, loss, name='weighted_loss')
        self.indiv_loss = loss
        return self.indiv_loss

    def train(self, feature_input, num_layers, num_classes, confidence, weak_label):
        global optimizer
        trainable_tvars, total_loss = self.aggregate_loss(feature_input, num_layers, num_classes, confidence, weak_label)
        with tf.variable_scope('optimize_tar_net'):

            # learning_rate = tf.divide(self.lr, feature_input.shape[1].value)
            learning_rate = self.lr
            grads = tf.gradients(total_loss, trainable_tvars, confidence)
            grads, _ = tf.clip_by_global_norm(grads, clip_norm=self.max_grad_norm)
            grad_var_pairs = zip(grads, trainable_tvars)

            if self.optimizer == 'sgd':
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate, name='sgd')
            elif self.optimizer == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name='adam')
            elif self.optimizer == 'adadelta':
                optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate, epsilon=1e-6, name='adadelta')
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

    def aggregate_loss(self, feature_input, num_layers, num_classes, confidence, weak_label):
        final_logits = self.predict_labels(feature_input, num_layers, num_classes)

        # get prediction
        self.logits = final_logits
        self.cnf_score = confidence
        self.probabilities = tf.nn.softmax(final_logits, name='softmax_probability')
        self.prediction = tf.cast(tf.argmax(input=self.probabilities, axis=1, name='prediction'), dtype=tf.int32)

        # loss = tf.reduce_mean(self.compute_loss(final_logits, weak_label, confidence))

        loss = self.compute_loss(final_logits, weak_label, confidence)

        global_tvars = tf.trainable_variables()

        # (Done) TODO: Check if target network params are const.
        cnf_net_tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "classifier/cnf_net")
        # rep_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "classifier/rep_layer")
        # emb_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "classifier/emb_lookup")
        trainable_tvars = list(set(global_tvars) - set(cnf_net_tvars))

        # (Done) TODO: Add regularization loss.
        l2_regularizer = tf.contrib.layers.l2_regularizer(scale=self.reg_const, scope="tar_reg")
        regularization_penalty = tf.contrib.layers.apply_regularization(l2_regularizer, trainable_tvars)
        self.total_loss = loss + regularization_penalty
        self.mean_loss = tf.reduce_mean(self.total_loss)
        return trainable_tvars, self.total_loss

    def assign_lr(self, session, lr_value):
        session.run(tf.assign(self.lr, lr_value))

    @property
    def lr(self):
        return self._lr
