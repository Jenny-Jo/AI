# RandomForest 적용
# cifar10 적용

import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score   # Kfold : 교차 검증
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC


# gridSearch
# 내가 정해놓은 조건들을 충족하는 것을 전부다 가져온다. 


#1. data
cancer = load_breast_cancer()

x = cancer.data         
y = cancer.target

print(x.shape)          # (569, 30)
print(y.shape)          # (569,)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 13, train_size =0.8)

parameters ={
    "n_estimators" : [100, 200],      
    "max_depth": [6, 8, 10, 20],      
    "min_samples_leaf":[3, 5, 7, 10], 
    "min_samples_split": [2,3, 5],     
    # 'max_features                   
    "n_jobs" : [-1]
}                                    

kfold = KFold(n_splits = 5, shuffle = True)                                                

model = RandomizedSearchCV(RandomForestClassifier(), parameters, cv =  kfold)  

model.fit(x_train, y_train)

print('최적의 매개변수 : ', model.best_estimator_)
y_pred = model.predict(x_test)
print("최종 정답률 = ", accuracy_score(y_test, y_pred) )