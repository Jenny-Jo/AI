# RandomiziedSearchCV
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.svm import SVC

#1. data
iris = load_iris()

x = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, 
                                                    shuffle = True, random_state = 43)


# grid / random search에서 사용할 매개 변수
parameters = [
    {'svm__C':[1, 10, 100, 1000], 'svm__kernel':['linear']},
    {'svm__C':[1, 10, 100, 1000], 'svm__kernel':['rbf'], 'svm__gamma':[0.001, 0.0001]},
    {'svm__C':[1, 10, 100, 1000], 'svm__kernel':['linear'], 'svm__gamma':[0.001, 0.0001]}
]

#2. model
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
pipe = Pipeline([("scaler", MinMaxScaler()), ('svm', SVC())])    


model = RandomizedSearchCV(pipe, parameters , cv = 5)

#3. fit
model.fit(x_train, y_train)


#4. evaluate, predict
acc = model.score(x_test, y_test)

print('최적의 매개변수 = ', model.best_estimator_)
print('acc : ', acc) # 할 때 마다 accuracy 표시