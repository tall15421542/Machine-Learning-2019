from sklearn.ensemble import RandomForestRegressor
import numpy as np 
class RandomForestPreprocessor:
    def __init__(self):
        self.model = RandomForestRegressor(criterion = "mae", n_jobs = -1)
        self.topk = -1

    def process(self, x_data, y_data = 0, topk = -1):
        if topk != -1: 
            self.topk = topk
            self.model.fit(x_data, y_data)
            self.topkIndex = np.argpartition(self.model.feature_importances_, -topk)[-topk:]
        return x_data[:, self.topkIndex]
    
