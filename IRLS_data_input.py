import numpy as np

class data_set(object):
    def __init__(self,file_path):
        self.file_path = file_path
        self.X_, self.Y_ = self.read_data_input()


    def read_data_input(self):
        f = open(self.file_path)
        lines = f.readlines()
        f.close()
        X_ = np.zeros((len(lines), 124))
        Y_ = np.zeros((len(lines),1))
        rows_num = 0
        for line in lines:
            tmp = line.split(' ')
            y_ = int(tmp[0])
            if(y_<0):
                y_ = 0
            Y_[rows_num] = y_
            for i in range(1, len(tmp) - 1):
                feature_tur = tmp[i].split(':')
                feature_id = feature_tur[0]
                feature_val = feature_tur[1]
                X_[rows_num, int(feature_id)-1] = float(feature_val)
            rows_num += 1
        X_[:,-1] = 1.0
        return X_,Y_
    # def K_fold_cross_validation