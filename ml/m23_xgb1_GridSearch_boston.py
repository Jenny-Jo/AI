from xgboost import XGBRegressor, plot_importance
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC


dataset = load_boston()
x = dataset.data
y = dataset.target

print(x.shape) # (506, 13)
print(y.shape) # (506,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8,
                                    shuffle = True, random_state = 66 )


# XGB 필수 파라미터 ##
n_estimators = 10000        #나무 100개/ decision tree보다 100배 느려
learning_rate = 0.001      # 학습률 디폴트 값/ 상당히 중요하다
colsample_bytree = 0.9
colsample_bylevel = 0.6   # 0.6~0.9
###
# 점수 :  0.9312397094966539

max_depth = 5
n_jobs = -1 #  딥러닝 빼고 default 로 써라

# model = XGBRegressor(max_depth= max_depth, learning_rate=learning_rate, n_estimators=n_estimators, n_jobs=n_jobs, 
#                        #colsample_bylevel = colsample_bylevel, 
#                        colsample_bytree= colsample_bytree)

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

score = model.score(x_test, y_test) # evaluate
print('점수 : ', score)

'''
-------------------------------------
XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=0.6,
             colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
             importance_type='gain', interaction_constraints='',
             learning_rate=0.1, max_delta_step=0, max_depth=6,
             min_child_weight=1, missing=nan, monotone_constraints='()',
             n_estimators=110, n_jobs=0, num_parallel_tree=1,
             objective='reg:squarederror', random_state=0, reg_alpha=0,
             reg_lambda=1, scale_pos_weight=1, subsample=1, tree_method='exact',
             validate_parameters=1, verbosity=None)
-------------------------------------
{'colsample_bylevel': 0.6, 'colsample_bytree': 1, 'learning_rate': 0.1, 'max_depth': 6, 'n_estimators': 110}
-------------------------------------
점수 :  0.9311752979545836
'''

