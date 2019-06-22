import numpy as np
class FinalSolver:
    def __init__(self):
        self.y_weight = np.array([1.0, 1.0, 1.0])

    def readXTrain(self, path):
        X_train = np.load(path)
        self.x_train = (X_train['arr_0'])
        X_train.close

    def readYTrain(self, path):
        Y_train = np.load(path)
        self.y_train = (Y_train['arr_0'])
        Y_train.close 

    def readXTest(self, path):
        X_test = np.load(path)
        self.x_test = (X_test['arr_0'])
        X_test.close

    def readXValidation(self, path):
        X_validation = np.load(path)
        self.x_val = (X_validation['arr_0'])
        X_validation.close

    def readYValidation(self, path):
        Y_validation = np.load(path)
        self.y_val = (Y_validation['arr_0'])
        Y_validation.close

    def setModel(self, model):
        self.model = model 

    def setPreprocessor(self, preprocessor):
        self.preprocessor = preprocessor 

    def preprocessData(self, topk):
        self.x_train = self.preprocessor.process(x_data = self.x_train, y_data = self.y_train * self.y_weight, topk = topk)

    def fit(self):
        self.model.fit(self.x_train, self.y_train * self.y_weight)
        self.model_y_weight = self.y_weight

    def scaleY(self, weight):
        self.y_weight = weight

    def predict(self, x_data):
        result = np.array(self.model.predict(x_data)) / self.model_y_weight;
        return result

    def outputPredict(self):
        pass 

    def setLossFunction(self, loss):
        self.loss = loss 

    def calculateEin(self):
        return self.calculateError(self.x_train, self.y_train)

    def calculateValError(self):
        return self.calculateError(self.x_val, self.y_val)

    def calculateError(self, x_data, y_data):
        if hasattr(self, 'preprocessor'):
            n, d = x_data.shape
            if d == 10000:
               x_data = self.preprocessor.process(x_data)

        return self.loss(self.predict(x_data), y_data)
