
'''
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
model.fit(x_train, y_train, verbose=True, eval_metric = 'rmse', eval_set = [(x_train, y_train), (x_test, y_test)])
# 회귀 평가지표 (rmse, mae) , logloss, error, auc

# fit한 것에 대한 평가
result = model.evals_result()
print('evals results: ',result)

y_pred = model.predict(x_test)

r2 = r2_score(y_pred, y_test)
print('r2 score : %.2f%%' %(r2*100.0))
print('r2:', r2)

'''
# xgboost evaluate

import numpy as np
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.datasets import load_boston, load_breast_cancer

## 데이터
# x, y = load_boston(return_X_y = True)
x, y = load_breast_cancer(return_X_y = True)

print(x.shape)          # (506, 13)
print(y.shape)          # (506,)

## train_test_split
x_train, x_test, y_train, y_test = train_test_split( x, y, test_size = 0.2,
    shuffle = True, random_state = 66)

## 모델링
model = XGBClassifier(n_estimators = 1000,        # verbose의 갯수, epochs와 동일
                     learning_rate = 0.1)

model.fit(x_train, y_train,
          verbose = True, eval_metric = 'rmse',
          eval_set = [(x_train, y_train), (x_test, y_test)])
# eval_metic의 종류 : rmse, mae, logloss, error(error가 0.2면 accuracy는 0.8), auc(정확도, 정밀도; accuracy의 친구다)

results = model.evals_result()
print("eval's result : ", results)

y_pred = model.predict(x_test)
acc = accuracy_score(y_pred, y_test)
print('acc: ', acc)

import pickle  # python에서 기본제공해서 바로 뜸
pickle.dump(model, open('F:/Study/model/sample/xgb_save/cancer.pickle.dat','wb'))
print('저장됨')

model2 = XGBClassifier()
model2 = pickle.load(open('F:/Study/model/sample/xgb_save/cancer.pickle.dat', 'rb'))
print('불러옴')

y_pred = model2.predict(x_test)
acc = accuracy_score(y_pred, y_test)
print('acc: ', acc)

