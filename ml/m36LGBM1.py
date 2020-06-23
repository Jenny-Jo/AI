# Light gradient boost machine
# XGB 보다 2년 늦게 나왔고, 속도 빠름
# 모듈 속도가 나오지 않음
# time 적용
# save 안되면 피클로 하기
from lightgbm import LGBMRegressor, LGBMClassifier
from xgboost import XGBRegressor, plot_importance  
from sklearn.feature_selection import SelectFromModel
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import numpy as np

# 회귀 모델
x, y = load_boston(return_X_y=True)

print(x.shape)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8,
                                                    shuffle = True, random_state = 66)

model = LGBMRegressor(n_estimators = 100, learning_rate = 0.05, n_jobs = -1) 

model.fit(x_train, y_train)

import time
start = time.time()

threshold = np.sort(model.feature_importances_)

for thres in threshold:
    selection = SelectFromModel(model, threshold = thres, prefit = True)
    
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)

    selection_model = LGBMRegressor(n_estimators = 10, learning_rate = 0.05, n_jobs = -1) 

    selection_model.fit(select_x_train, y_train, verbose= True, eval_metric= ['logloss', 'rmse'],
                                        eval_set= [(select_x_train, y_train), (select_x_test, y_test)],
                                        early_stopping_rounds= 20)

    y_pred = selection_model.predict(select_x_test)
    r2 = r2_score(y_test, y_pred)
    print('R2 : ', r2)
     
    print("Thresh=%.3f, n = %d, R2 : %.2f%%" %(thres, select_x_train.shape[1], r2*100.0))

    # result = selection_model.evals_result()
    # print("eval's result : ", result)

# R2 :  0.9354279986548603


end = time.time() - start
print('총 걸린 시간1 :',end)

# 1)
import pickle  # python에서 기본제공해서 바로 뜸
pickle.dump(model, open('F:/Study/model/sample/lgbm_Save/boston.pickle.dat_lgbm','wb')) # wb write binary 

# # 2)
# import joblib 
# joblib.dump(model, 'F:/Study/model/sample/lgbm_Save/boston.joblib_Thresh=0.569, n = 1, R2 : -172.39%_.dat')

# 3)
# model.save_model('F:/Study/model/sample/lgbm_Save/boston.xgb.model_Thresh=0.569, n = 1, R2 : -172.39%_')
#xgb
print('저장됨')


model2 = LGBMRegressor()
# model2 = pickle.load(open('F:/Study/model/sample/lgbm_Save/boston.pickle_R2.dat', 'rb')) # rb read binary
# model2 = joblib.load('F:/Study/model/sample/lgbm_Save/boston.joblib.dat')
# model2.load_model('F:/Study/model/sample/lgbm_Save/boston.xgb.model') # R2 :  -1.7238583749182492

print('불러옴')