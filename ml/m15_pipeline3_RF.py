# RandomiziedSearchCV
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

#1. data
iris = load_iris()

x = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, 
                                                    shuffle = True, random_state = 43)

parameters = [
    {'randomforestclassifier__n_jobs':[1]} #, 10, 100, 1000], 'randomforestclassifier__kernel':['linear']},
    # {'randomforestclassifier__C':[1, 10, 100, 1000], 'randomforestclassifier__kernel':['rbf'], 'randomforestclassifier__gamma':[0.001, 0.0001]},
    # {'randomforestclassifier__C':[1, 10, 100, 1000], 'randomforestclassifier__kernel':['linear'], 'randomforestclassifier__gamma':[0.001, 0.0001]}
]

# grid / random search에서 사용할 매개 변수/ 리스트 형태의 키:밸류 딕셔너리
# parameters = [
#     {'svm__C':[1, 10, 100, 1000], 'svm__kernel':['linear']},
#     {'svm__C':[1, 10, 100, 1000], 'svm__kernel':['rbf'], 'svm__gamma':[0.001, 0.0001]},
#     {'svm__C':[1, 10, 100, 1000], 'svm__kernel':['linear'], 'svm__gamma':[0.001, 0.0001]}
# ]
# # under bag 2개 + 모델명 명시를 해줘야 에러 안뜬다##




#2. model
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# pipe = Pipeline([("scaler", MinMaxScaler()), ('svm', SVC())])  
#                               모델, 전처리  
                                            # 여기 있는 모델명을 명시해줘야 함##
pipe = make_pipeline(MinMaxScaler(), RandomForestClassifier())
                                    # 여기랑 같아야
model = RandomizedSearchCV(pipe, parameters , cv = 5)


# make pipeline

#3. fit
model.fit(x_train, y_train)


#4. evaluate, predict
acc = model.score(x_test, y_test)

print('최적의 매개변수 = ', model.best_estimator_)
print('acc : ', acc) # 할 때 마다 accuracy 표시