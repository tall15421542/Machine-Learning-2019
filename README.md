# Machine Learning 2019
## solver
### Usage
```
python3 Main.py [N_data] [Data_Preprocessor] [model] [writePredict]
```
#### N_data: int
size for trainning data and validation data 

if ``N_data == -1`` 

Use ``X_train.npz``, ``Y_train.npz`` as trainning data 

#### Data_Preprocessor 
preprocessor's registered name 

if ``Data_Preprocessor == 'n'`` 

``Data_preprocessor`` is not used

#### model 
model's registered name

#### writePredict
if `` writePredict == "w"`` 

write test's predict under ``predict/`` folder 

if preprocessor == "n" 

file's name is ``{model's name}-predict.csv``  

else 

file's name  is ``{preprocessor's name}-{model's name}-predict.csv`` 


### add new model
1. create class with ``predict`` and ``fit`` method implementation under ``model/`` 
```python
class Model:
    def fit(self, x_data, y_data):
        ...
    def predict(self, x_data):
        ...
        return predict_y_data
```

2. register in ``Main.py``
```python
model_option = {
    "svr": SVMR(3),
    "randomforest": RandomForestRegressor(criterion = "mae", n_jobs = -1)
    newModelName: Model()
}

```
### add new preprocessor
1. create class with ``process`` method implementation under ``preprocessor/``
```python
class Preprocessor:
    def process(self, x_data, y_data = 0, topk = 0):
        # case: preprocess in training stage
        # trainning and assign self.{preprocessor} 
        # return processed x_data
        
        # case: preprocess in predict stage
        # return processed x_data directly
```

2. register in ``Main.py``
```python
preprocessor_option = {
        "randomforest": RandomForestPreprocessor(),
        "pca": PCAPreprocessor(),
        PreprocessorName: Preprocessor()
}
```
### Example usage under ``solver/``
``python3 Main.py 100 randomforest randomforest n``  

``python3 Main.py 100 n svr w``
