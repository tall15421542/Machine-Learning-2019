from sklearn.ensemble import RandomForestRegressor
import numpy as np 
class RandomForestPreprocessor:
    def __init__(self):
        self.model = RandomForestRegressor(
                criterion = "mae", n_jobs = -1, n_estimators = 100, min_samples_leaf = 0.05,
                max_features = "sqrt", verbose = True, warm_start = True, oob_score = True)
        self.topk = -1

    def process(self, x_data, y_data = 0, topk = -1):
        if topk != -1: 
            self.topk = topk
            self.model.fit(x_data, y_data)
            self.topkIndex = np.argpartition(self.model.feature_importances_, -topk)[-topk:]
        return x_data[:, self.topkIndex]

    def report(self):
        print(self.model.oob_score_)
        argIdx = np.argsort(self.model.feature_importances_)[-self.topk:]
        print(self.model.feature_importances_[argIdx])
    
