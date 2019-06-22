import numpy as np 
import random
import sys
import re

class SmallerDataCreator():
    def create(self, path):
        print("Creating smaller data...")
        N_data = int(re.findall(r'\d+', path)[0])
        # x_train : [47500,10000]
        X_train = np.load('./dataset/X_train.npz')
        x_train = X_train['arr_0']

        # random unique N_data length list
        n, d = x_train.shape 
        randIdx = random.sample(range(0, n), N_data * 2)
        train_idx = randIdx[:N_data]
        val_idx = randIdx[N_data:]
        print(train_idx)
        print(val_idx)

        np.savez("./dataset/X_train_small_{}".format(N_data), x_train[train_idx])

        # x_validate
        np.savez("./dataset/X_validation_{}".format(N_data), x_train[val_idx])
        X_train.close

        # y_train
        Y_train = np.load('./dataset/Y_train.npz')
        y_train = Y_train['arr_0']
        np.savez("./dataset/Y_train_small_{}".format(N_data), y_train[train_idx])

        # y_validate
        np.savez("./dataset/Y_validation_{}".format(N_data), y_train[val_idx])
        Y_train.close

        print("Create Done")


