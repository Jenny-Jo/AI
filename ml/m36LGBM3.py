from lightgbm import LGBMRegressor, LGBMClassifier
from xgboost import XGBClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

x, y = load_iris(return_X_y=True)

print(x.shape)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8,
                                                    shuffle = True, random_state = 66)

model = LGBMClassifier(objective='multi:softmax', n_estimators = 100, learning_rate = 0.05, n_jobs = -1)

model.fit(x_train, y_train)

import time
start = time.time()

threshold = np.sort(model.feature_importances_)

for thres in threshold:
    selection = SelectFromModel(model, threshold = thres, prefit = True)
    
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)

    selection_model = LGBMClassifier(objective='multi:softmax', n_estimators = 100, learning_rate = 0.05, n_jobs = -1) 

    selection_model.fit(select_x_train, y_train, verbose= True, eval_metric= ['multi_logloss', 'multi_error'],
                                        eval_set= [(select_x_train, y_train), (select_x_test, y_test)],
                                        early_stopping_rounds= 20)

    y_pred = selection_model.predict(select_x_test)
    acc = accuracy_score(y_test, y_pred)
     
    print("Thresh=%.3f, n = %d, ACC : %.2f%%" %(thres, select_x_train.shape[1], acc*100.0))

    # result = selection_model.evals_result()
    # print("eval's result : ", result)

# Thresh=0.644, n = 1, ACC : 100.00%

end = time.time() - start
print('총 걸린 시간 : ', end)

# 1)
import pickle  # python에서 기본제공해서 바로 뜸
pickle.dump(model, open('F:/Study/model/sample/lgbm_Save/iris.pickle.dat_lgbm','wb')) # wb write binary 

# # 2)
# import joblib 
# joblib.dump(model, 'F:/Study/model/sample/lgbm_Save/iris.joblib_Thresh=0.569, n = 1, R2 : -172.39%_.dat')

# 3)
# model.save_model('F:/Study/model/sample/lgbm_Save/iris.xgb.model_Thresh=0.569, n = 1, R2 : -172.39%_')
#xgb
print('저장됨')


model2 = LGBMClassifier()
model2 = pickle.load(open('F:/Study/model/sample/lgbm_Save/iris.pickle.dat_lgbm', 'rb')) # rb read binary
# model2 = joblib.load('F:/Study/model/sample/lgbm_Save/iris.joblib.dat')
# model2.load_model('F:/Study/model/sample/lgbm_Save/iris.xgb.model') # R2 :  -1.7238583749182492

print('불러옴')

# acc :  0.7192982456140351
# Thresh=201.000, n = 1, ACC : 71.93%