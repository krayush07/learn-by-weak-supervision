import os
import random
import sys
import time

import numpy as np
import tensorflow as tf

from global_module.implementation_module import L2LWS, DataReader
from global_module.settings_module import GlobalParams, ConfidenceNetworkParams, TargetNetworkParams, \
    Dictionary, Directory, ConfidenceNetworkDirectory, TargetNetworkDirectory


class Train:
    def __init__(self):
        self.iter_cnf_train = 0
        self.iter_cnf_valid = 0
        self.iter_tar_train = 0
        self.iter_cnf_valid = 0

    def run_cnf_net_epoch(self, writer, session, min_cost, model_obj, dict_obj, epoch_num):
        global total_loss
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

            if model_obj.cnf_params.mode == 'TR':
                self.iter_cnf_train += 1
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary, indiv_loss, total_loss, tg, lg, sb, one_hot, fg, cg, rep, f, fc_op, _ = session.run([net_obj.merged_train,
                                                                                                              net_obj.indiv_loss,
                                                                                                              net_obj.total_loss,
                                                                                                              net_obj.tg,
                                                                                                              net_obj.lg,
                                                                                                              net_obj.sb,
                                                                                                              net_obj.one_hot,
                                                                                                              net_obj.final_lg,
                                                                                                              net_obj.cnf_sc,
                                                                                                              model_obj.cnf_rep,
                                                                                                              net_obj.feat_input,
                                                                                                              net_obj.fc_op,
                                                                                                              model_obj.cnf_train_op],
                                                                                                             options=run_options,
                                                                                                             run_metadata=run_metadata,
                                                                                                             feed_dict=feed_dict)

                if params.log:
                    writer.add_run_metadata(run_metadata, 'step%d' % self.iter_cnf_train)
                    writer.add_summary(summary, self.iter_cnf_train)

            else:
                indiv_loss, total_loss, tg, lg, sb, one_hot, fg, cg, rep, f, fc_op, _ = session.run([net_obj.indiv_loss,
                                                                                                     net_obj.total_loss,
                                                                                                     net_obj.tg,
                                                                                                     net_obj.lg,
                                                                                                     net_obj.sb,
                                                                                                     net_obj.one_hot,
                                                                                                     net_obj.final_lg,
                                                                                                     net_obj.cnf_sc,
                                                                                                     model_obj.cnf_rep,
                                                                                                     net_obj.feat_input,
                                                                                                     net_obj.fc_op,
                                                                                                     model_obj.cnf_train_op],
                                                                                                    feed_dict=feed_dict)
            epoch_combined_loss += total_loss

        print 'Epoch Num: %d, CE loss: %.4f' % (epoch_num + 1, epoch_combined_loss)

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

    def run_tar_net_epoch(self, writer, session, min_cost, model_obj, dict_obj, epoch_num):
        net_obj = model_obj.tar_network
        params = model_obj.tar_params
        dir_obj = model_obj.tar_dir
        data_filename = dir_obj.data_filename
        gold_label_filename = dir_obj.gold_label_filename
        weak_label_filename = dir_obj.weak_label_filename

        correct = 0.
        total = 0.

        epoch_combined_loss = 0.0
        for step, (data_arr, gold_label_arr, weak_label_arr) \
                in enumerate(DataReader(params).tar_data_iterator(data_filename,
                                                                  gold_label_filename,
                                                                  weak_label_filename,
                                                                  model_obj.tar_params.indices,
                                                                  dict_obj)):

            # a = gold_label_arr
            # b = np.zeros((params.batch_size, params.num_classes))
            # b[np.arange(params.batch_size), a] = 1

            feed_dict = {model_obj.unlabeled_text: data_arr,
                         model_obj.weak_label_unlabeled: weak_label_arr
                         # model_obj.weak_label_unlabeled: b
                         }

            if model_obj.tar_params.mode == 'TR':
                self.iter_tar_train += 1
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()

                summary, indiv_loss, total_loss, pred, prob, logits, cnf, rep, _ = session.run([net_obj.merged_train,
                                                                                                net_obj.indiv_loss,
                                                                                                net_obj.mean_loss,
                                                                                                net_obj.prediction,
                                                                                                net_obj.probabilities,
                                                                                                net_obj.logits,
                                                                                                net_obj.cnf_score,
                                                                                                model_obj.tar_rep,
                                                                                                model_obj.tar_train_op],
                                                                                               options=run_options,
                                                                                               run_metadata=run_metadata,
                                                                                               feed_dict=feed_dict)

                if params.log:
                    writer.add_run_metadata(run_metadata, 'step%d' % self.iter_tar_train)
                    writer.add_summary(summary, self.iter_tar_train)


            else:
                indiv_loss, total_loss, pred, prob, logits, cnf, rep, _ = session.run([net_obj.indiv_loss,
                                                                                       net_obj.mean_loss,
                                                                                       net_obj.prediction,
                                                                                       net_obj.probabilities,
                                                                                       net_obj.logits,
                                                                                       net_obj.cnf_score,
                                                                                       model_obj.tar_rep,
                                                                                       model_obj.tar_train_op],
                                                                                      feed_dict=feed_dict)
            for idx, each_pred in enumerate(pred):
                if each_pred == gold_label_arr[idx]:
                    correct += 1

            total += params.batch_size

            epoch_combined_loss += total_loss

        print('Accuracy: %.4f' % (correct / total))
        print 'Epoch Num: %d, CE loss: %.4f' % (epoch_num + 1, epoch_combined_loss)

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
        tf.set_random_seed(1234)
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
        cnf_params_valid.batch_size = 64

        # tar valid object
        tar_params_valid = TargetNetworkParams(mode=mode_valid)
        tar_dir_valid = TargetNetworkDirectory(mode_valid)
        tar_params_valid.num_instances, tar_params_valid.indices = self.get_length(tar_dir_valid.data_filename)
        tar_params_valid.batch_size = 64

        # params_train.num_classes = params_valid.num_classes = len(dict_obj.label_dict)

        if global_params.enable_shuffle:
            random.shuffle(cnf_params_train.indices)
            random.shuffle(cnf_params_valid.indices)
            random.shuffle(tar_params_train.indices)
            random.shuffle(tar_params_valid.indices)

        cnf_min_loss = tar_min_loss = sys.float_info.max

        word_emb_path = global_dir.word_embedding
        word_emb_matrix = np.float32(np.genfromtxt(word_emb_path, delimiter=' '))
        global_params.vocab_size = len(word_emb_matrix)

        print('***** INITIALIZING TF GRAPH *****')

        timestamp = str(int(time.time()))
        # cnf_train_out_dir = os.path.abspath(os.path.join(dir_train.log_path, "train", timestamp))
        # valid_out_dir = os.path.abspath(os.path.join(dir_train.log_path, "valid", timestamp))
        # print("Writing to {}\n".format(cnf_train_out_dir))

        cnf_train_out_dir = os.path.abspath(os.path.join(global_dir.cnf_log_path, "train", timestamp))
        cnf_valid_out_dir = os.path.abspath(os.path.join(global_dir.cnf_log_path, "valid", timestamp))

        tar_train_out_dir = os.path.abspath(os.path.join(global_dir.tar_log_path, "train", timestamp))
        tar_valid_out_dir = os.path.abspath(os.path.join(global_dir.tar_log_path, "valid", timestamp))

        with tf.Graph().as_default(), tf.Session() as session:

            # random_normal_initializer = tf.random_normal_initializer()
            random_uniform_initializer = tf.random_uniform_initializer(-global_params.init_scale, global_params.init_scale)
            # xavier_initializer = tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32)

            with tf.variable_scope("classifier", reuse=None, initializer=random_uniform_initializer):
                train_obj = L2LWS(global_params,
                                  cnf_params_train,
                                  tar_params_train,
                                  global_dir,
                                  cnf_dir_train,
                                  tar_dir_train)

            cnf_train_writer = tf.summary.FileWriter(cnf_train_out_dir, session.graph)
            cnf_valid_writer = tf.summary.FileWriter(cnf_valid_out_dir)

            tar_train_writer = tf.summary.FileWriter(tar_train_out_dir, session.graph)
            tar_valid_writer = tf.summary.FileWriter(tar_valid_out_dir)

            if not global_params.enable_checkpoint:
                session.run(tf.global_variables_initializer())

            if global_params.enable_checkpoint:
                ckpt = tf.train.get_checkpoint_state(global_dir.model_path)
                if ckpt and ckpt.model_checkpoint_path:
                    print("Loading model from: %s" % ckpt.model_checkpoint_path)
                    tf.train.Saver().restore(session, ckpt.model_checkpoint_path)
            elif not global_params.use_random_initializer:
                session.run(tf.assign(train_obj.word_emb_matrix, word_emb_matrix, name="word_embedding_matrix"))

            with tf.variable_scope("classifier", reuse=True, initializer=random_uniform_initializer):
                valid_obj = L2LWS(global_params,
                                  cnf_params_valid,
                                  tar_params_valid,
                                  global_dir,
                                  cnf_dir_valid,
                                  tar_dir_valid)

            print('**** TF GRAPH INITIALIZED ****')

            start_time = time.time()

            for i in range(global_params.max_max_epoch):

                print('\n++++++++=========+++++++\n')

                if i >= global_params.pre_train_epoch:
                    ckpt = tf.train.get_checkpoint_state(global_dir.model_path)
                    if ckpt and ckpt.model_checkpoint_path:
                        print("Loading model from: %s" % ckpt.model_checkpoint_path)
                        tf.train.Saver().restore(session, ckpt.model_checkpoint_path)

                lr_decay = cnf_params_train.lr_decay ** max(i - global_params.max_epoch, 0.0)
                train_obj.cnf_network.assign_lr(session, cnf_params_train.learning_rate * lr_decay)
                train_obj.tar_network.assign_lr(session, cnf_params_train.learning_rate * lr_decay)

                print("Epoch: %d Learning rate: %.5f" % (i + 1, session.run(train_obj.cnf_network.lr)))
                train_loss, _ = self.run_cnf_net_epoch(cnf_train_writer, session, cnf_min_loss, train_obj, dict_obj, i)
                print("CONFIDENCE NETWORK: Epoch: %d Train loss: %.3f\n" % (i + 1, train_loss))

                valid_obj.cnf_network.assign_lr(session, cnf_params_train.learning_rate * lr_decay)
                valid_obj.tar_network.assign_lr(session, cnf_params_train.learning_rate * lr_decay)

                valid_loss, curr_loss = self.run_cnf_net_epoch(cnf_valid_writer, session, cnf_min_loss, valid_obj, dict_obj, i)
                if curr_loss < cnf_min_loss:
                    cnf_min_loss = curr_loss

                print("CONFIDENCE NETWORK: Epoch: %d Valid loss: %.3f" % (i + 1, valid_loss))

                curr_time = time.time()
                print('1 epoch run takes ' + str(((curr_time - start_time) / (i + 1)) / 60) + ' minutes.')

                print('\n++++++++=========+++++++\n')

                ''' TARGET NETWORK '''

                if i >= global_params.pre_train_epoch:

                    if ckpt and ckpt.model_checkpoint_path:
                        print("Loading model from: %s" % ckpt.model_checkpoint_path)
                        tf.train.Saver().restore(session, ckpt.model_checkpoint_path)

                    lr_decay = tar_params_train.lr_decay ** max(i - global_params.max_epoch, 0.0)
                    train_obj.cnf_network.assign_lr(session, cnf_params_train.learning_rate * lr_decay)
                    train_obj.tar_network.assign_lr(session, cnf_params_train.learning_rate * lr_decay)

                    print("Epoch: %d Learning rate: %.5f" % (i + 1, session.run(train_obj.tar_network.lr)))
                    train_loss, _ = self.run_tar_net_epoch(tar_train_writer, session, tar_min_loss, train_obj, dict_obj, i)
                    print("TARGET NETWORK: Epoch: %d Train loss: %.3f\n" % (i + 1, train_loss))

                    valid_obj.cnf_network.assign_lr(session, cnf_params_train.learning_rate * lr_decay)
                    valid_obj.tar_network.assign_lr(session, cnf_params_train.learning_rate * lr_decay)

                    valid_loss, curr_loss = self.run_tar_net_epoch(tar_valid_writer, session, tar_min_loss, valid_obj, dict_obj, i)
                    if curr_loss < tar_min_loss:
                        tar_min_loss = curr_loss

                    print("TARGET NETWORK: Epoch: %d Valid loss: %.3f" % (i + 1, valid_loss))

                    curr_time = time.time()
                    print('1 epoch run takes ' + str(((curr_time - start_time) / (i + 1)) / 60) + ' minutes.')

                    print('\n++++++++=========+++++++\n')

            cnf_train_writer.close()
            cnf_valid_writer.close()
            tar_train_writer.close()
            tar_valid_writer.close()


def main():
    dict_obj = Dictionary()
    Train().run_train(dict_obj)


if __name__ == "__main__":
    main()
