import os
import random
import sys
import time

import numpy as np
import tensorflow as tf

from global_module.implementation_module import L2LWS, DataReader
from global_module.settings_module import GlobalParams, ConfidenceNetworkParams, TargetNetworkParams, \
    Dictionary, Directory, ConfidenceNetworkDirectory, TargetNetworkDirectory

iter_train = 0
iter_valid = 0


class Train:
    def run_cnf_net_epoch(self, session, min_cost, model_obj, dict_obj, epoch_num):
        net_obj = model_obj.cnf_network
        params = model_obj.cnf_params
        dir_obj = model_obj.cnf_dir
        data_filename = dir_obj.data_filename
        gold_label_filename = dir_obj.gold_label_filename
        weak_label_filename = dir_obj.weak_label_filename

        epoch_combined_loss = 0.0
        for step, (data_arr, gold_label_arr, weak_label_arr) \
                in enumerate(DataReader(params).cnf_data_iterator(data_filename, gold_label_filename,
                                                                  weak_label_filename,
                                                                  model_obj.cnf_params.indices,
                                                                  dict_obj)):
            feed_dict = {model_obj.labeled_text: data_arr,
                         model_obj.gold_label: gold_label_arr,
                         model_obj.weak_label_labeled: weak_label_arr
                         }

            indiv_loss, total_loss, _ = session.run([net_obj.indiv_loss,
                                                     net_obj.total_loss,
                                                     # net_obj.tg,
                                                     # net_obj.lg,
                                                     # net_obj.sb,
                                                     # net_obj.one_hot,
                                                     model_obj.cnf_train_op],
                                                    feed_dict=feed_dict)

            epoch_combined_loss += total_loss

        print 'Epoch Num: %d, CE loss: %.4f' % (epoch_num, epoch_combined_loss)

        if params.mode == 'VA':
            model_saver = tf.train.Saver()
            print('**** Current minimum on valid set: %.4f ****' % min_cost)

            if epoch_combined_loss < min_cost:
                min_cost = epoch_combined_loss
                model_saver.save(session,
                                 save_path=dir_obj.model_path + dir_obj.model_name,
                                 latest_filename=dir_obj.latest_checkpoint)
                print('==== Model saved! ====')

        return epoch_combined_loss, min_cost

    def run_tar_net_epoch(self, session, min_cost, model_obj, dict_obj, epoch_num):
        net_obj = model_obj.tar_net
        params = model_obj.tar_params
        dir_obj = model_obj.tar_dir
        data_filename = dir_obj.data_filename
        weak_label_filename = dir_obj.weak_label_filename

        epoch_combined_loss = 0.0
        for step, (data_arr, weak_label_arr) \
                in enumerate(DataReader(params).tar_data_iterator(data_filename,
                                                                  weak_label_filename,
                                                                  model_obj.tar_params.indices,
                                                                  dict_obj)):
            feed_dict = {model_obj.labeled_text: data_arr,
                         model_obj.weak_label_labeled: weak_label_arr
                         }

            loss, _ = session.run([net_obj.loss,
                                   model_obj.run_confidence_network],
                                  feed_dict=feed_dict)

            epoch_combined_loss += loss

        print 'Epoch Num: %d, CE loss: %.4f' % (epoch_num, epoch_combined_loss)

        if params.mode == 'VA':
            model_saver = tf.train.Saver()
            print('**** Current minimum on valid set: %.4f ****' % min_cost)

            if epoch_combined_loss < min_cost:
                min_cost = epoch_combined_loss
                model_saver.save(session,
                                 save_path=dir_obj.model_path + dir_obj.model_name,
                                 latest_filename=dir_obj.latest_checkpoint)
                print('==== Model saved! ====')

        return epoch_combined_loss, min_cost

    def get_length(self, filename):
        print('Reading :', filename)
        data_file = open(filename, 'r')
        count = 0
        for _ in data_file:
            count += 1
        data_file.close()
        return count, np.arange(count)

    def run_train(self, dict_obj):
        mode_train, mode_valid, mode_test = 'TR', 'VA', 'TE'

        # global params and dir
        global_params = GlobalParams()
        global_dir = Directory('TR')

        # cnf train object
        cnf_params_train = ConfidenceNetworkParams(mode=mode_train)
        cnf_dir_train = ConfidenceNetworkDirectory(mode_train)
        cnf_params_train.num_instances, cnf_params_train.indices = self.get_length(cnf_dir_train.data_filename)

        # tar train object
        tar_params_train = TargetNetworkParams(mode=mode_train)
        tar_dir_train = TargetNetworkDirectory(mode_train)
        tar_params_train.num_instances, tar_params_train.indices = self.get_length(tar_dir_train.data_filename)

        # cnf valid object
        cnf_params_valid = ConfidenceNetworkParams(mode=mode_valid)
        cnf_dir_valid = ConfidenceNetworkDirectory(mode_valid)
        cnf_params_valid.num_instances, cnf_params_valid.indices = self.get_length(cnf_dir_valid.data_filename)
        cnf_params_valid.batch_size = 2

        # tar valid object
        tar_params_valid = TargetNetworkParams(mode=mode_valid)
        tar_dir_valid = TargetNetworkDirectory(mode_valid)
        tar_params_valid.num_instances, tar_params_valid.indices = self.get_length(tar_dir_valid.data_filename)
        tar_params_valid.batch_size = 2

        # params_train.num_classes = params_valid.num_classes = len(dict_obj.label_dict)

        if global_params.enable_shuffle:
            random.shuffle(cnf_params_train.indices)
            random.shuffle(cnf_params_valid.indices)
            random.shuffle(tar_params_train.indices)
            random.shuffle(tar_params_valid.indices)

        min_loss = sys.float_info.max

        word_emb_path = global_dir.word_embedding
        word_emb_matrix = np.float32(np.genfromtxt(word_emb_path, delimiter=' '))
        global_params.vocab_size = len(word_emb_matrix)

        print('***** INITIALIZING TF GRAPH *****')

        timestamp = str(int(time.time()))
        # train_out_dir = os.path.abspath(os.path.join(dir_train.log_path, "train", timestamp))
        # valid_out_dir = os.path.abspath(os.path.join(dir_train.log_path, "valid", timestamp))
        # print("Writing to {}\n".format(train_out_dir))

        with tf.Graph().as_default(), tf.Session() as session:

            # random_normal_initializer = tf.random_normal_initializer()
            # random_uniform_initializer = tf.random_uniform_initializer(-params_train.init_scale, params_train.init_scale)
            xavier_initializer = tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32)

            with tf.variable_scope("classifier", reuse=None, initializer=xavier_initializer):
                train_obj = L2LWS(global_params,
                                  cnf_params_train,
                                  tar_params_train,
                                  global_dir,
                                  cnf_dir_train,
                                  tar_dir_train)

            # train_writer = tf.summary.FileWriter(train_out_dir, session.graph)
            # valid_writer = tf.summary.FileWriter(valid_out_dir)

            if not global_params.enable_checkpoint:
                session.run(tf.global_variables_initializer())

            if global_params.enable_checkpoint:
                ckpt = tf.train.get_checkpoint_state(global_dir.model_path)
                if ckpt and ckpt.model_checkpoint_path:
                    print("Loading model from: %s" % ckpt.model_checkpoint_path)
                    tf.train.Saver().restore(session, ckpt.model_checkpoint_path)
            elif not global_params.use_random_initializer:
                session.run(tf.assign(train_obj.word_emb_matrix, word_emb_matrix, name="word_embedding_matrix"))

            with tf.variable_scope("classifier", reuse=True, initializer=xavier_initializer):
                valid_obj = L2LWS(global_params,
                                  cnf_params_valid,
                                  tar_params_valid,
                                  global_dir,
                                  cnf_dir_valid,
                                  tar_dir_valid)

            print('**** TF GRAPH INITIALIZED ****')

            start_time = time.time()
            for i in range(global_params.max_max_epoch):

                lr_decay = cnf_params_train.lr_decay ** max(i - global_params.max_epoch, 0.0)
                train_obj.cnf_network.assign_lr(session, cnf_params_train.learning_rate * lr_decay)
                train_obj.tar_network.assign_lr(session, cnf_params_train.learning_rate * lr_decay)

                # print(params_train.learning_rate * lr_decay)

                print('\n++++++++=========+++++++\n')

                # print("Epoch: %d Learning rate: %.5f" % (i + 1, session.run(train_obj.lr)))
                train_loss, _ = self.run_cnf_net_epoch(session, min_loss, train_obj, dict_obj, i)
                print("Epoch: %d Train loss: %.3f" % (i + 1, train_loss))

                valid_obj.cnf_network.assign_lr(session, cnf_params_train.learning_rate * lr_decay)
                valid_obj.tar_network.assign_lr(session, cnf_params_train.learning_rate * lr_decay)

                valid_loss, curr_loss = self.run_cnf_net_epoch(session, min_loss, valid_obj, dict_obj, i)
                if curr_loss < min_loss:
                    min_loss = curr_loss

                print("Epoch: %d Valid loss: %.3f" % (i + 1, valid_loss))

                curr_time = time.time()
                print('1 epoch run takes ' + str(((curr_time - start_time) / (i + 1)) / 60) + ' minutes.')

                # train_writer.close()
                # valid_writer.close()


def main():
    dict_obj = Dictionary()
    Train().run_train(dict_obj)


if __name__ == "__main__":
    main()
