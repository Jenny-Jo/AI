'''
컴퓨터 과학에서 파이프라인(영어: pipeline)은 
한 데이터 처리 단계의 출력이 다음 단계의 입력으로 이어지는 형태로 연결된 구조를 가리킨다. 
이렇게 연결된 데이터 처리 단계는 한 여러 단계가 서로 동시에, 
또는 병렬적으로 수행될 수 있어 효율성의 향상을 꾀할 수 있다.

http://www.itworld.co.kr/news/105167

모델, 파라미터, CV 가 뭐의 구성요소?
'''
import pandas as pd
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC 

# 1. Data
iris = load_iris()

x = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, shuffle=True, random_state=43)

# 2. model
# model = SVC()
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

pipe = Pipeline([("scaler", MinMaxScaler()),('SVM', SVC())]) # model  과 전처리 방식 쓰겠다

pipe.fit(x_train, y_train)
print("acc: ", pipe.score(x_test, y_test))



