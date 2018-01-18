import os
from global_module.settings_module import Directory


class ConfidenceNetworkDirectory(Directory):
    def __init__(self, mode):
        Directory.__init__(self, mode)

        self.data_path += '/cnf'

        ''' DATASET '''
        self.raw_data_filename = self.data_path + self.raw_data_filename
        self.data_filename = self.data_path + self.data_filename
        self.gold_label_filename = self.data_path + self.gold_label_filename
        self.weak_label_filename = self.data_path + self.weak_label_filename

    def makedir(self, dirname):
        if not os.path.exists(dirname):
            os.makedirs(dirname)
