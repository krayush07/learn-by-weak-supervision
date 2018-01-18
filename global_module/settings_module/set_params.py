class GlobalParams:
    def __init__(self, mode='TR'):
        """
        :param mode: 'TR' for train, 'TE' for test, 'VA' for valid
        """
        self.mode = mode
        self.init_scale = 0.1
        self.learning_rate = 0.01
        self.max_grad_norm = 10
        self.max_epoch = 300
        self.max_max_epoch = 500

        if (mode == 'TR'):
            self.keep_prob = 0.5
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

        self.REG_CONSTANT = 0.00001
        self.EMB_DIM = 300
        self.MAX_LEN = 120
        self.num_hidden_layer = 3

        self.optimizer = 'sgd'

        self.batch_size = 64
        self.vocab_size = 30
        self.is_word_trainable = False

        self.use_unknown_word = True
        self.use_random_initializer = False

        self.indices = None
        self.num_instances = None
        self.num_classes = 20
        self.sampling_threshold = 2

        ''' PARAMS FOR CONV BLOCK '''
        self.num_filters = 256
        self.filter_width = [3, 4, 5]
        self.conv_activation = 'RELU'
        self.conv_padding = 'VALID'

        self.pool_width = 10
        self.pool_stride = 3
        self.pool_padding = 'VALID'
        self.pool_option = 'MAX'
        self.if_pool_max = True # if pool width is equal to convoluted matrix

        self.pre_train_epoch = 5