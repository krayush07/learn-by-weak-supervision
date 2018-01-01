import tensorflow as tf

from global_module.settings_module import GlobalParams, ConfidenceNetworkParams, TargetNetworkParams, Directory
from global_module.implementation_module import RepLayer, ConfidenceNetwork, TargetNetwork


class L2LWS:
    def __init__(self, global_params, cnf_params, tar_params, dir_obj, cnf_dir_obj, tar_dir_obj):
        self.global_params = global_params
        self.cnf_params = cnf_params
        self.tar_params = tar_params
        self.dir_obj = dir_obj
        self.cnf_dir = cnf_dir_obj
        self.tar_dir = tar_dir_obj
        self.init_pipeline()

    def init_pipeline(self):
        self.create_placeholders()
        self.extract_word_embedding()
        self.init_representation_layer()
        self.init_target_network()
        self.init_confidence_network()
        self.run_confidence_network()
        self.run_target_network()

    def create_placeholders(self):
        self.labeled_text = tf.placeholder(dtype=tf.int32,
                                           shape=[None, self.global_params.MAX_LEN],
                                           name='labeled_txt_placeholder')

        self.unlabeled_text = tf.placeholder(dtype=tf.int32,
                                             shape=[None, self.global_params.MAX_LEN],
                                             name='unlabeled_txt_placeholder')

        self.gold_label = tf.placeholder(dtype=tf.int32,
                                         shape=[None],
                                         name='gold_label_placeholder')

        self.weak_label_labeled = tf.placeholder(dtype=tf.float32,
                                                 shape=[None, self.global_params.num_classes],
                                                 name='weak_labeled_placeholder')

        self.weak_label_unlabeled = tf.placeholder(dtype=tf.float32,
                                                   shape=[None, self.global_params.num_classes],
                                                   name='weak_unlabeled_placeholder')

    def extract_word_embedding(self):
        with tf.variable_scope('emb_lookup'):
            self.word_emb_matrix = tf.get_variable("word_embedding_matrix",
                                                   shape=[self.global_params.vocab_size, self.global_params.EMB_DIM],
                                                   dtype=tf.float32,
                                                   regularizer=tf.contrib.layers.l2_regularizer(0.0),
                                                   trainable=self.global_params.is_word_trainable)

            self.labeled_word_emb = tf.nn.embedding_lookup(params=self.word_emb_matrix,
                                                           ids=self.labeled_text,
                                                           name='labeled_word_emb',
                                                           validate_indices=True)

            self.unlabeled_word_emb = tf.nn.embedding_lookup(params=self.word_emb_matrix,
                                                             ids=self.unlabeled_text,
                                                             name='unlabeled_word_emb',
                                                             validate_indices=True)
            print 'Extracted word embedding'

    def init_representation_layer(self):
        with tf.variable_scope('rep_layer'):
            self.rep_layer = RepLayer()

    def init_confidence_network(self):
        with tf.variable_scope('cnf_net'):
            self.cnf_network = ConfidenceNetwork(self.global_params.num_classes, self.cnf_params.optimizer, self.cnf_params.max_grad_norm)

    def run_confidence_network(self):
        with tf.variable_scope('cnf_net'):
            if self.cnf_params.mode == 'TR':
                run_cnf = self.cnf_network.train(self.rep_layer.create_representation(self.labeled_word_emb, self.cnf_params),
                                       num_layers=3,
                                       true_label=self.gold_label,
                                       weak_label=self.weak_label_labeled)
            else:
                run_cnf, _ = self.cnf_network.aggregate_loss(self.rep_layer.create_representation(self.labeled_word_emb, self.cnf_params),
                                                num_layers=3,
                                                true_label=self.gold_label,
                                                weak_label=self.weak_label_labeled)
            self.cnf_train_op = run_cnf

    def init_target_network(self):
        with tf.variable_scope('tar_net'):
            self.tar_network = TargetNetwork(self.tar_params.optimizer, self.tar_params.max_grad_norm, self.tar_params.REG_CONSTANT)

    def run_target_network(self):
        with tf.variable_scope('cnf_net', reuse=True):
            confidence = self.cnf_network.compute_confidence(self.rep_layer.create_representation(self.unlabeled_word_emb, self.global_params), 3)
        with tf.variable_scope('tar_net'):
            if self.tar_params.mode == 'TR':
                run_tar = self.tar_network.train(self.rep_layer.create_representation(self.unlabeled_word_emb, self.global_params),
                                       num_layers=3,
                                       num_classes=self.global_params.num_classes,
                                       confidence=confidence,
                                       weak_label=self.weak_label_unlabeled)
            else:
                run_tar = self.tar_network.aggregate_loss(self.rep_layer.create_representation(self.unlabeled_word_emb, self.global_params),
                                                num_layers=3,
                                                num_classes=self.global_params.num_classes,
                                                confidence=confidence,
                                                weak_label=self.weak_label_unlabeled)

            self.tar_train_op = run_tar

# def main():
#     L2LWS(GlobalParams(), ConfidenceNetworkParams(), TargetNetworkParams(), Directory('TR'))
#
#
# if __name__ == '__main__':
#     main()
