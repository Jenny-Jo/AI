

from sklearn.feature_selection import SelectFromModel
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, r2_score
# dataset = load_boston()
# x = dataset.data
# y = dataset.target

x, y = load_iris(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8,
                                                    shuffle = True, random_state = 66)

model = XGBRegressor()
model.fit(x_train, y_train)
score = model.score(x_test, y_test)
print('acc : ', score)

# feature engineering
thresholds = np.sort(model.feature_importances_)
print(thresholds)
# 오름차순으로 중요도가 낮은 애들부터 나옴

# R2 :  0.9221188544655419
# [0.00134153 0.00363372 0.01203115 0.01220458 0.01447935 0.01479119
#  0.0175432  0.03041655 0.04246345 0.0518254  0.06949984 0.30128643
#  0.42848358]

import time
start = time.time()

# 전체 컬럼수 13개만큼 돌림
for thresh in thresholds : # thresh : feature importance 값 // 모델 선택함
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
                                                # median
    select_x_train = selection.transform(x_train)
    
    # print(select_x_train.shape)

# colomn이 하나씩, 중요하지 않은 애들부터 지움
# 중요한 애들 빼내

    selection_model =  XGBRegressor()
    selection_model.fit(select_x_train, y_train)      
    
    select_x_test = selection.transform(x_test)
    y_pred = selection_model.predict(select_x_test)
    
    score = r2_score(y_test, y_pred)
    print('acc : ', score)
    
    print('Thresh =%.3f, n=%d, R2 : %.2f%%' %(thresh,select_x_train.shape[1],
                                              score*100))

end = time.time() - start
print('총 걸린 시간1 :',end)

import time
start2 = time.time()

# 전체 컬럼수 13개만큼 돌림
for thresh in thresholds : # thresh : feature importance 값 // 모델 선택함
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
                                                # median
    select_x_train = selection.transform(x_train)
    
    # print(select_x_train.shape)

# colomn이 하나씩, 중요하지 않은 애들부터 지움
# 중요한 애들 빼내

    selection_model =  XGBRegressor(n_jobs = 2, n_estimators=1000)
    selection_model.fit(select_x_train, y_train)      
    
    select_x_test = selection.transform(x_test)
    y_pred = selection_model.predict(select_x_test)
    
    score = r2_score(y_test, y_pred)
    print('acc : ', score)
    
    print('Thresh =%.3f, n=%d, R2 : %.2f%%' %(thresh,select_x_train.shape[1],
                                              score*100))

end2 = time.time() - start
print('총 걸린 시간2 :',end2)
