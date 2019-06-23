import lightgbm as lgb 
import numpy as np 

class LGBM:
    def __init__(self):
        self.params = []
        self.params.append({
                'num_leaves': 150,
                'max_bin': 255,
                'min_data_in_leaf': 1,
                'learning_rate': 0.02,
                'bagging_fraction': 1.0, 
                'bagging_freq': 5, 
                'feature_fraction': 1.0,
                'min_gain_to_split': 0.65,
                'max_depth': 10,
                'save_binary': True,
                'seed': 1337,
                'feature_fraction_seed': 1337,
                'bagging_seed': 1337,
                'data_random_seed': 1337,
                'application': 'regression_l1',
                'boosting_type': 'gbdt',
                #'boosting_type': 'dart',

                'verbose': -1,
                'metric': {'l1'},
            })

        self.params.append({
                'num_leaves': 500,
                'max_bin': 255,
                'min_data_in_leaf': 100,
                'learning_rate': 0.1,
                'bagging_fraction': 1,
                'bagging_freq': 2,
                'feature_fraction': 0.8,
                #'min_gain_to_split': 0.65,
                'max_depth': 10,
                'save_binary': True,
                'seed': 1337,
                'feature_fraction_seed': 1337,
                'bagging_seed': 1337,
                'data_random_seed': 1337,
                'application': 'regression_l1',
                'boosting_type': 'gbdt',
                #'lambda_l1':5,
                #'boosting_type': 'dart',

                'verbose': -1,
                'metric': {'l1'},
            })

        self.params.append( {
                'num_leaves': 500,
                'max_bin': 255,
                'min_data_in_leaf': 1,
                'learning_rate': 0.1,
                'bagging_fraction': 1,
                'bagging_freq': 2,
                'feature_fraction': 1,
                #'min_gain_to_split': 0.65,
                'max_depth': 15,
                'save_binary': True,
                'seed': 1337,
                'feature_fraction_seed': 1337,
                'bagging_seed': 1337,
                'data_random_seed': 1337,
                'application': 'regression_l1',
                'boosting_type': 'gbdt',
                #'lambda_l1':5,
                #'boosting_type': 'dart',

                'verbose': -1,
                'metric': {'l1'},
            })
        self.gbms = []

    def fit(self, x_data, y_data):
        lgb_train = []
        lgb_train.append(lgb.Dataset(x_data,y_data[:, 0]))
        lgb_train.append(lgb.Dataset(x_data,y_data[:, 1]))
        lgb_train.append(lgb.Dataset(x_data,y_data[:, 2]))
        self.gbms.append(lgb.train(self.params[0],lgb_train[0],num_boost_round=1500))
        self.gbms.append(lgb.train(self.params[1],lgb_train[1],num_boost_round=2500))
        self.gbms.append(lgb.train(self.params[2],lgb_train[2],num_boost_round=2500))

    def predict(self, x_data):
        predict_y = []
        for gbm in self.gbms:
            predict_y.append(gbm.predict(x_data, num_iteration=gbm.best_iteration))
        return np.array(predict_y).transpose() 
