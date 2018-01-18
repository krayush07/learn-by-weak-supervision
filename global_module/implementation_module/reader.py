import numpy as np
import math
import random
from sklearn.utils import shuffle


class DataReader:
    def __init__(self, params):
        self.params = params

    def get_index_string(self, utt, word_dict):
        index_string = ''
        for each_token in utt.split():
            if (self.params.all_lowercase):
                if (word_dict.has_key(each_token.lower())):
                    each_token = each_token.lower()
                elif (word_dict.has_key(each_token)):
                    each_token = each_token
                elif (word_dict.has_key(each_token.title())):
                    each_token = each_token.title()
                elif (word_dict.has_key(each_token.upper())):
                    each_token = each_token.upper()
                else:
                    each_token = each_token.lower()

            index_string += str(word_dict.get(each_token, word_dict.get("UNK"))) + '\t'
        return len(index_string.strip().split()), index_string.strip()

    def pad_string(self, id_string, curr_len, max_seq_len):
        id_string = id_string.strip() + '\t'
        while curr_len < max_seq_len:
            id_string += '0\t'
            curr_len += 1
        return id_string.strip()

    def format_string(self, inp_string, curr_string_len, max_len):
        if curr_string_len > max_len:
            # print('Maximum SEQ LENGTH reached. Stripping extra sequence.\n')
            op_string = '\t'.join(inp_string.split('\t')[:max_len])
        else:
            op_string = self.pad_string(inp_string, curr_string_len, max_len)
        return op_string

    def generate_cnf_id_map(self, data_filename, gold_label_filename, weak_label_filename, index_arr, dict_obj):
        data_file_arr = open(data_filename, 'r').readlines()
        glabel_file_arr = open(gold_label_filename, 'r').readlines()
        wlabel_file_arr = open(weak_label_filename, 'r').readlines()

        global_data_arr = []
        global_glabel_arr = []
        global_wlabel_arr = []

        for curr_id, each_idx in enumerate(index_arr):
            curr_line = data_file_arr[each_idx].strip()
            curr_glabel = int(glabel_file_arr[each_idx].strip())
            curr_wlabel = map(np.float32, wlabel_file_arr[each_idx].strip().split('\t'))
            string_len, index_string = self.get_index_string(curr_line, dict_obj.word_dict)
            curr_line_index_string = self.format_string(index_string, string_len, self.params.MAX_LEN)  # format resp string

            global_data_arr.append(map(int, curr_line_index_string.split('\t')))
            global_glabel_arr.append(curr_glabel)
            global_wlabel_arr.append(curr_wlabel)

        return global_data_arr, global_glabel_arr, global_wlabel_arr

    def generate_tar_id_map(self, data_filename, weak_label_filename, index_arr, dict_obj):
        data_file_arr = open(data_filename, 'r').readlines()
        wlabel_file_arr = open(weak_label_filename, 'r').readlines()

        global_data_arr = []
        global_wlabel_arr = []

        for curr_id, each_idx in enumerate(index_arr):
            curr_line = data_file_arr[each_idx].strip()
            curr_wlabel = map(np.float32, wlabel_file_arr[each_idx].strip().split('\t'))
            string_len, index_string = self.get_index_string(curr_line, dict_obj.word_dict)
            curr_line_index_string = self.format_string(index_string, string_len, self.params.MAX_LEN)  # format resp string

            global_data_arr.append(map(int, curr_line_index_string.split('\t')))
            global_wlabel_arr.append(curr_wlabel)

        return global_data_arr, global_wlabel_arr

    def cnf_data_iterator(self, data_filename, gold_label_filename, weak_label_filename, index_arr, dict_obj):
        data_arr, gold_label_arr, weak_label_arr = self.generate_cnf_id_map(data_filename,
                                                                            gold_label_filename,
                                                                            weak_label_filename,
                                                                            index_arr,
                                                                            dict_obj)

        batch_size = self.params.batch_size
        num_batches = len(index_arr) / self.params.batch_size

        if self.params.mode == 'TR':
            num_batches = int(math.ceil(0.4 * num_batches))
            # random.shuffle(index_arr)
            data_arr, gold_label_arr, weak_label_arr = shuffle(data_arr, gold_label_arr, weak_label_arr)

        for i in range(num_batches):
            curr_data_arr = data_arr[i * batch_size: (i + 1) * batch_size]
            curr_gold_label_arr = gold_label_arr[i * batch_size: (i + 1) * batch_size]
            curr_weak_label_arr = weak_label_arr[i * batch_size: (i + 1) * batch_size]

            # for p in range(batch_size):
            yield (np.array(curr_data_arr), np.array(curr_gold_label_arr), np.array(curr_weak_label_arr))

    # def tar_data_iterator(self, data_filename, weak_label_filename, index_arr, dict_obj):
    #     data_arr, weak_label_arr = self.generate_tar_id_map(data_filename,
    #                                                         weak_label_filename,
    #                                                         index_arr,
    #                                                         dict_obj)
    #     batch_size = self.params.batch_size
    #     num_batches = len(index_arr) / self.params.batch_size
    #
    #     if self.params.mode == 'TR':
    #         num_batches = int(math.ceil(0.3 * num_batches))
    #         # random.shuffle(index_arr)
    #         data_arr, weak_label_arr = shuffle(data_arr, weak_label_arr)
    #
    #     for i in range(num_batches):
    #         curr_data_arr = data_arr[i * batch_size: (i + 1) * batch_size]
    #         curr_weak_label_arr = weak_label_arr[i * batch_size: (i + 1) * batch_size]
    #
    #         # for p in range(batch_size):
    #         yield (np.array(curr_data_arr), np.array(curr_weak_label_arr))


    def tar_data_iterator(self, data_filename, gold_label_filename, weak_label_filename, index_arr, dict_obj):
        data_arr, gold_label_arr, weak_label_arr = self.generate_cnf_id_map(data_filename,
                                                                            gold_label_filename,
                                                                            weak_label_filename,
                                                                            index_arr,
                                                                            dict_obj)

        batch_size = self.params.batch_size
        num_batches = len(index_arr) / self.params.batch_size

        if self.params.mode == 'TR':
            num_batches = int(math.ceil(0.5 * num_batches))
            # random.shuffle(index_arr)
            data_arr, gold_label_arr, weak_label_arr = shuffle(data_arr, gold_label_arr, weak_label_arr)

        for i in range(num_batches):
            curr_data_arr = data_arr[i * batch_size: (i + 1) * batch_size]
            curr_gold_label_arr = gold_label_arr[i * batch_size: (i + 1) * batch_size]
            curr_weak_label_arr = weak_label_arr[i * batch_size: (i + 1) * batch_size]

            # for p in range(batch_size):
            yield (np.array(curr_data_arr), np.array(curr_gold_label_arr), np.array(curr_weak_label_arr))