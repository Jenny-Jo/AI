# boston으로 모델링 # r2_score만 추가
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.utils.testing import all_estimators
import warnings

warnings.filterwarnings('ignore')

boston = load_boston()
x = boston.data
y = boston.target

print(x.shape) # (506, 13)
print(y.shape) # (506, )

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=44, shuffle = True) #shuffle true 가 default

allAlgorithms = all_estimators(type_filter = 'regressor')

for (name, algorithm) in allAlgorithms:
    model = algorithm()

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print(name, "의 정답률 =", r2_score(y_test, y_pred))

import sklearn
print(sklearn.__version__)
'''
ARDRegression 의 정답률 = 0.7512651671065589        
AdaBoostRegressor 의 정답률 = 0.8483607422925512    
BaggingRegressor 의 정답률 = 0.8856054507736202     
BayesianRidge 의 정답률 = 0.7444777786443533        
CCA 의 정답률 = 0.7270542664211517
DecisionTreeRegressor 의 정답률 = 0.8344741531629991
ElasticNet 의 정답률 = 0.699050089875551
ElasticNetCV 의 정답률 = 0.6902681369495265
ExtraTreeRegressor 의 정답률 = 0.7529229719828645   
ExtraTreesRegressor 의 정답률 = 0.8857420503368705  
GaussianProcessRegressor 의 정답률 = -5.639147690233129
GradientBoostingRegressor 의 정답률 = 0.8948035509660103
HuberRegressor 의 정답률 = 0.7043100781423643       
KNeighborsRegressor 의 정답률 = 0.6390759816821279  
KernelRidge 의 정답률 = 0.7744886784070626
Lars 의 정답률 = 0.7521800808693163
LarsCV 의 정답률 = 0.7521800808693163
Lasso 의 정답률 = 0.6855879495660049
LassoCV 의 정답률 = 0.71540574604873
LassoLars 의 정답률 = -0.0007982049217318821        
LassoLarsCV 의 정답률 = 0.7521800808693163
LassoLarsIC 의 정답률 = 0.7540945959884459
LinearRegression 의 정답률 = 0.752180080869314      
LinearSVR 의 정답률 = 0.6759566249734841
MLPRegressor 의 정답률 = 0.3286990911163856

sklearn 0.20.1에서 제공하는 모든 모델이다
'''