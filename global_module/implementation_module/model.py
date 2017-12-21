import tensorflow as tf

from global_module.settings_module import ParamsClass, Directory
from global_module.implementation_module import RepLayer, ConfidenceNetwork, TargetNetwork


class L2LWS:
    def __init__(self, params, dir_obj):
        self.params = params
        self.dir_obj = dir_obj
        self.init_pipeline()

    def init_pipeline(self):
        self.create_placeholders()
        self.extract_word_embedding()
        self.init_representation_layer()
        self.init_target_network()
        self.init_confidence_network()

        self.train_confidence_network()
        self.train_target_network()

    def create_placeholders(self):
        self.labeled_text = tf.placeholder(dtype=tf.int32,
                                           shape=[None, self.params.MAX_LEN],
                                           name='labeled_txt_placeholder')

        self.unlabeled_text = tf.placeholder(dtype=tf.int32,
                                             shape=[None, self.params.MAX_LEN],
                                             name='unlabeled_txt_placeholder')

        self.gold_label = tf.placeholder(dtype=tf.int32,
                                         shape=[None],
                                         name='gold_label_placeholder')

        self.weak_label_labeled = tf.placeholder(dtype=tf.float32,
                                                 shape=[None, self.params.num_classes],
                                                 name='weak_labeled_placeholder')

        self.weak_label_unlabeled = tf.placeholder(dtype=tf.float32,
                                                   shape=[None, self.params.num_classes],
                                                   name='weak_unlabeled_placeholder')

    def extract_word_embedding(self):
        with tf.variable_scope('emb_lookup'):
            self.word_emb_matrix = tf.get_variable("word_embedding_matrix",
                                                   shape=[self.params.vocab_size, self.params.EMB_DIM],
                                                   dtype=tf.float32,
                                                   regularizer=tf.contrib.layers.l2_regularizer(0.0),
                                                   trainable=self.params.is_word_trainable)

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
            self.cnf_network = ConfidenceNetwork(self.params.num_classes, self.params.cnf_optimizer, self.params.max_grad_norm)

    def train_confidence_network(self):
        with tf.variable_scope('cnf_net'):
            self.cnf_network.train(self.rep_layer.create_representation(self.labeled_word_emb, self.params),
                                   3,
                                   self.gold_label,
                                   self.weak_label_labeled,
                                   lr=0.03)

    def init_target_network(self):
        with tf.variable_scope('tar_net'):
            self.tar_network = TargetNetwork(self.params.tar_optimizer, self.params.max_grad_norm)

    def train_target_network(self):
        with tf.variable_scope('cnf_net', reuse=True):
            confidence = self.cnf_network.compute_confidence(self.rep_layer.create_representation(self.labeled_word_emb, self.params), 3)
        with tf.variable_scope('tar_net'):
            self.tar_network.train(self.rep_layer.create_representation(self.unlabeled_word_emb, self.params),
                                   3, self.params.num_classes, confidence, self.weak_label_unlabeled,
                                   lr=0.03)


def main():
    L2LWS(ParamsClass(), Directory('TR'))


if __name__ == '__main__':
    main()
