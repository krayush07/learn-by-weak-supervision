class GlobalParams:
    def __init__(self, mode='TR'):
        """
        :param mode: 'TR' for train, 'TE' for test, 'VA' for valid
        """
        self.mode = mode
        self.init_scale = 0.1
        self.learning_rate = 0.01
        self.max_grad_norm = 10
        self.max_epoch = 100
        self.max_max_epoch = 200

        if (mode == 'TR'):
            self.keep_prob = 0.6
        else:
            self.keep_prob = 1.0

        self.lr_decay = 0.99

        self.enable_shuffle = True
        self.enable_checkpoint = False
        self.all_lowercase = True
        self.log = False
        self.log_step = 9

        if (mode == 'TE'):
            self.enable_shuffle = False

        self.REG_CONSTANT = 0.01
        self.MAX_SEQ_LEN = 20
        self.EMB_DIM = 300
        self.MAX_LEN = 20

        self.optimizer = 'sgd'

        self.batch_size = 2
        self.vocab_size = 30
        self.is_word_trainable = True

        self.use_unknown_word = True
        self.use_random_initializer = False

        self.indices = None
        self.num_instances = None
        self.num_classes = 3
        self.sampling_threshold = 2

        ''' PARAMS FOR CONV BLOCK '''
        self.num_filters = 128
        self.filter_width = [2, 3, 5]
        self.conv_activation = 'RELU'
        self.conv_padding = 'VALID'

        self.pool_width = 10
        self.pool_stride = 3
        self.pool_padding = 'VALID'
        self.pool_option = 'MAX'
