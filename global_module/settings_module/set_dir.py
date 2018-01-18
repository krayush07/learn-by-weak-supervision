import os


class Directory():
    def __init__(self, mode):
        """
        :param mode: 'TR' for train, 'TE' for test, 'VA' for valid
        """

        self.mode = mode

        self.root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        self.utility_dir = self.root_path + '/utility_dir'
        self.curr_utility_dir = self.utility_dir + '/folder1'
        self.preprocessing_dir = self.root_path + '/pre_processing'

        '''Directory to utility paths'''
        self.data_path = self.curr_utility_dir + '/data'
        self.vocab_path = self.curr_utility_dir + '/vocab'
        self.model_path = self.curr_utility_dir + '/models'
        self.output_path = self.curr_utility_dir + '/output'
        self.log_path = self.curr_utility_dir + '/log_dir'
        self.cnf_log_path = self.log_path + '/cnf'
        self.tar_log_path = self.log_path + '/tar'

        self.makedir(self.vocab_path)
        self.makedir(self.model_path)
        self.makedir(self.output_path)
        self.makedir(self.log_path)
        self.makedir(self.cnf_log_path)
        self.makedir(self.tar_log_path)

        # self.glove_path = '/data/glove_dir/glove_300_pickle' + '/glove_dict.pkl'
        self.glove_path = '/home/aykumar/aykumar_home/glove_dir' + '/glove_dict.pkl'

        '''Directory to dataset'''

        self.raw_base_file = self.data_path + '/raw_base_file.txt'

        self.raw_train_path = '/raw_tokenized_train.txt'

        self.raw_data_filename = '/raw_data_train.txt'
        self.data_filename = '/tokenized_train.txt'
        self.gold_label_filename = '/gold_label_train.txt'
        self.weak_label_filename = '/weak_label_train.txt'

        if (mode == 'VA'):
            self.raw_data_filename = '/raw_data_valid.txt'
            self.data_filename = '/tokenized_valid.txt'
            self.gold_label_filename = '/gold_label_valid.txt'
            self.weak_label_filename = '/weak_label_valid.txt'
        elif (mode == 'TE'):
            self.raw_data_filename = '/raw_data_test.txt'
            self.data_filename = '/tokenized_test.txt'
            self.gold_label_filename = '/gold_label_test.txt'
            self.weak_label_filename = '/weak_label_test.txt'

        ''' ****************** Directory to saving or loading a model ********************** '''''
        self.latest_checkpoint = 'checkpoint'
        self.model_name = '/model.ckpt'  # model name .ckpt is the model extension
        ''' ********** ********* ******** ********* ********* ********* ******** ************* '''''

        self.test_cost_path = self.output_path + '/test_cost.txt'  # test cost output
        self.test_pred_path = self.output_path + '/test_pred.txt'
        self.test_seq_op_path = self.output_path + '/test_seq_op.txt'

        '''Directory to csv and pkl files'''
        self.vocab_size_file = self.vocab_path + '/vocab_size.txt'
        self.word_embedding = self.vocab_path + '/word_embedding.csv'
        self.word_vocab_dict = self.vocab_path + '/word_vocab.pkl'
        self.glove_present_training_word_vocab = self.vocab_path + '/glove_present_training_word_vocab.pkl'
        self.label_map_dict = self.vocab_path + '/label_to_id_dict.pkl'

        ''' ****************** Directory for test model ********************** '''''
        self.test_model_name = '/model.ckpt'
        self.test_model = self.model_path + self.test_model_name

    def makedir(self, dirname):
        if not os.path.exists(dirname):
            os.makedirs(dirname)
