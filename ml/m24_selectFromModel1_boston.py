

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

parameters = [
    { 'n_estimators': [100, 200, 300], 'learning_rate': [0.1, 0.3, 0.5, 0.01],
     'max_depth':[4,5,6] },
    
    { 'n_estimators': [90, 100, 110], 'learning_rate': [0.1, 0.001, 0.01],
     'max_depth':[4,5,6], 'colsample_bytree' : [0.6 ,0.9 ,1] },
    
    { 'n_estimators': [90, 110], 'learning_rate': [0.1, 0.001, 0.5],
     'max_depth':[4,5,6], 'colsample_bytree' : [0.6 ,0.9 ,1],
     'colsample_bylevel' : [0.6, 0.7, 0.9] }

]
n_jobs = -1

model = GridSearchCV(XGBRegressor(), parameters, cv=5, n_jobs=-1) # n_jobs 는 속도 높이기 위해 그리드 서치에 통으로 돌린다

model.fit(x_train, y_train)
print('-------------------------------------')
print(model.best_estimator_)
print('-------------------------------------')
print(model.best_params_)
print('-------------------------------------')


score = model.score(x_test, y_test)
print('R2 : ', score)


#### feature engineering ####

thresholds = np.sort(model.feature_importances_)
print(thresholds)
# 오름차순으로 중요도가 낮은 애들부터 나옴

# R2 :  0.9221188544655419
# [0.00134153 0.00363372 0.01203115 0.01220458 0.01447935 0.01479119
#  0.0175432  0.03041655 0.04246345 0.0518254  0.06949984 0.30128643
#  0.42848358]

# 전체 컬럼수 13개만큼 돌림
for thresh in thresholds : # thresh : feature importance 값 // 모델 선택함
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
                                                # median
    select_x_train = selection.transform(x_train)
    
    # print(select_x_train.shape)
'''
(404, 13)
(404, 12)
(404, 11)
(404, 10)
(404, 9)
(404, 8)
(404, 7)
(404, 6)
(404, 5)
(404, 4)
(404, 3)
(404, 2)
(404, 1)
'''
# colomn이 하나씩, 중요하지 않은 애들부터 지움
# 중요한 애들 빼내

    selection_model =  XGBRegressor()
    selection_model.fit(select_x_train, y_train)      
    
    select_x_test = selection.transform(x_test)
    y_pred = selection_model.predict(select_x_test)
    
    score = r2_score(y_test, y_pred)
    print('R2 : ', score)
    
    print('Thresh =%.3f, n=%d, R2 : %.2f%%' %(thresh,select_x_train.shape[1],
                                              score*100))
'''
Thresh =0.001, n=13, R2 : 92.21%
R2 :  0.921593214554105
Thresh =0.004, n=12, R2 : 92.16%
R2 :  0.9203131446000834
Thresh =0.012, n=11, R2 : 92.03%
R2 :  0.9219134406317994
Thresh =0.012, n=10, R2 : 92.19%
R2 :  0.9307724255990869
Thresh =0.014, n=9, R2 : 93.08%
R2 :  0.9236679494937362
Thresh =0.015, n=8, R2 : 92.37%
R2 :  0.9148067494950485
Thresh =0.018, n=7, R2 : 91.48%
R2 :  0.9270688746921889
Thresh =0.030, n=6, R2 : 92.71%
R2 :  0.9173564889681338
Thresh =0.042, n=5, R2 : 91.74%
R2 :  0.9210956498765558
Thresh =0.052, n=4, R2 : 92.11%
R2 :  0.925174592739333
Thresh =0.069, n=3, R2 : 92.52%
R2 :  0.694098937345699
Thresh =0.301, n=2, R2 : 69.41%
R2 :  0.44984470683830424
Thresh =0.428, n=1, R2 : 44.98%
'''

# Grid Search  까지 엮기
# Dacon 적용하기. 71개 컬럼