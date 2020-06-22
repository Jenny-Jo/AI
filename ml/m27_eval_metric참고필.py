

from sklearn.feature_selection import SelectFromModel
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.datasets import load_boston
from sklearn.metrics import accuracy_score, r2_score
# dataset = load_boston()
# x = dataset.data
# y = dataset.target

x, y = load_boston(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8,
                                                    shuffle = True, random_state = 66)

model = XGBRegressor(n_estimators =400, learning_rate = 0.1)
model.fit(x_train, y_train, verbose=True, eval_metric = ['logloss,','rmse'], eval_set = [(x_train, y_train), (x_test, y_test)],
          early_stopping_rounds=20)
# 회귀 평가지표 (rmse, mae) , logloss, error, auc

# fit한 것에 대한 평가
result = model.evals_result()
print('evals results: ',result)


y_pred = model.predict(x_test)

r2 = r2_score(y_pred, y_test)
print('r2 score : %.2f%%' %(r2*100.0))
print('r2:', r2)

