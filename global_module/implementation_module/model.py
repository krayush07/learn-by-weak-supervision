import tensorflow as tf

from global_module.settings_module import ParamsClass, Directory
from global_module.implementation_module import RepLayer

class L2LWS:
    def __init__(self, params, dir_obj):
        self.params = params
        self.dir_obj = dir_obj
        self.init_pipeline()

    def init_pipeline(self):
        self.create_placeholders()
        self.extract_word_embedding()

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
                                                 shape=[None],
                                                 name='weak_labeled_placeholder')

        self.weak_label_unlabeled = tf.placeholder(dtype=tf.float32,
                                                   shape=[None],
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
                                                             ids=self.unlabeled_word_emb,
                                                             name='unlabeled_word_emb',
                                                             validate_indices=True)
            print 'Extracted word embedding'

    def init_representation_layer(self):
        with tf.variable_scope('rep_layer'):
            self.rep_layer = RepLayer()