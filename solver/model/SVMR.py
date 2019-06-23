from sklearn.svm import SVR
from itertools import product
import numpy as np

class SVMR():
    def __init__(self, n_dim):
        self.model = []
        for i in range(n_dim):
            self.model.append(SVR(kernel='linear'))

    def fit(self, x_data, y_data):
        for index, svr in enumerate(self.model):
            svr.fit(x_data, y_data[:, index])

    def predict(self, x_data):
        predict_y = []
        for svr in self.model:
            predict_y.append(svr.predict(x_data))
        return np.array(predict_y).transpose() 
    
    def report(self):
        for idx, model in enumerate(self.model):
            print("Dim[{}], Number of SV: {}".format(idx, len(model.support_vectors_)))

