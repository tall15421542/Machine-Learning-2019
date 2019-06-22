import sys 
sys.path.insert(0, './preprocessor')
sys.path.insert(0, './model')

# import preprocessor 
from randomForestPreprocessor import RandomForestPreprocessor
from PCAPreprocessor import PCAPreprocessor

# import model 
from sklearn.ensemble import RandomForestRegressor
from SVMR import SVMR

# required import 
from finalSolver import FinalSolver
from itertools import product
from math import fabs
import numpy as np 

N_data = int(sys.argv[1])
selected_preprocessor = sys.argv[2]
selected_model = sys.argv[3]
writePredict = sys.argv[4]

preprocessor_option = {
        "randomforest": RandomForestPreprocessor(),
        "pca": PCAPreprocessor(),
}

model_option = {
    "svr": SVMR(3),
    "randomforest": RandomForestRegressor(criterion = "mae", n_jobs = -1)
}

# loss function 
def WMAE(y, y_hat):
    n, d = y.shape
    ret = 0
    weight = [300, 1, 200]
    for n_idx, d_idx in product(range(n), range(d)):
        ret += weight[d_idx] * fabs(y[n_idx][d_idx] - y_hat[n_idx][d_idx] ) 
    return ret / n


def NAE(y, y_hat):
    n, d = y.shape
    ret = 0
    for n_idx, d_idx in product(range(n), range(d)):
        ret += fabs(y[n_idx][d_idx] - y_hat[n_idx][d_idx] ) / y[n_idx][d_idx]
    return ret / n


if __name__ == '__main__':
    N_selected_feature = 100 

    solver = FinalSolver()
    if N_data != -1:
        solver.readXTrain("./dataset/X_train_small_{}.npz".format(N_data))
        solver.readYTrain("./dataset/Y_train_small_{}.npz".format(N_data))
        solver.readXValidation("./dataset/X_validation_{}.npz".format(N_data))
        solver.readYValidation("./dataset/Y_validation_{}.npz".format(N_data))
        solver.readXTest("./dataset/X_test.npz")
    else: 
        solver.readXTrain("./dataset/X_train.npz")
        solver.readYTrain("./dataset/Y_train.npz")

    print("trainning shape")
    print("x: {}".format(solver.x_train.shape))
    print("y: {}".format(solver.y_train.shape))
    print("\n###\n")

    # preprocess
    if selected_preprocessor != "n":
        solver.scaleY(np.array([300.0, 1.0, 200.0]))
        solver.setPreprocessor(selected_preprocessor, preprocessor_option[selected_preprocessor])
        solver.preprocessData(N_selected_feature)
        print("x_train shape after preprocess: {}".format(solver.x_train.shape))
        print("preprocessDone")
        print("\n###\n")

    # model train
    solver.setModel(selected_model, model_option[selected_model])
    solver.fit()
    print("\n###\n")

    # error 
    # WMAE 
    solver.setLossFunction(WMAE)
    print("WMAE EIN: {}".format(solver.calculateEin()))
    if N_data != -1:
        print("Validation: {}".format(solver.calculateValError()))
        print("\n###\n")

    # NAE
    solver.setLossFunction(NAE)
    print("NAE EIN: {}".format(solver.calculateEin()))
    if N_data != -1:
        print("Validation: {}".format(solver.calculateValError()))
        print("\n###\n")

    if writePredict != None:
        solver.writePredict(solver.x_test)

