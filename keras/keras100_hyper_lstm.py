# 97번을 RandomizedSearchCV로 변경하시오
'''
랜덤서치와 그리드서치 차이

 일반적인 parameter(weight)는 학습 과정에서 조정됩니다. 
 그러나 hyper-parameter(ex. Learning rate, Batch size 등)는 고정된 값으로 학습되며, 학습 이전에 우리가 지정해야 합니다. 
 데이터 셋마다 최적의 hyper-parameter가 다르고, 
 더 좋은 모델도 hyper-parameter 최적화 없이는 안좋은 결과를 얻기 때문에 hyper-parameter 최적화는 매우 중요합니다.
대표적인 hyper-parameter 최적화 방법은 Manual Search, Grid Search, Random Search, Bayesian Search가 있습니다. 
먼저 Random Search는 중요한 hyper-parameter를 더 많이 탐색합니다. 

이처럼, hyper-parameter는 상대적 중요도가 서로 다르고 Random Search는 중요한 parameter를 더 많이 탐색할 수 있기 때문에 
최적화하기에 유리합니다.
 반면, Grid Search는 중요하지 않은 hyper-parameter를 너무 많이 탐색한다고 합니다.

https://shwksl101.github.io/ml/dl/2019/01/30/Hyper_parameter_optimization.html
'''

from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Input
import numpy as np



#1. data
(x_train, y_train),(x_test, y_test) = mnist.load_data()

print(x_train.shape)                                   # (60000, 28, 28)
print(x_test.shape)                                    # (10000, 28, 28)

x_train = x_train.reshape(x_train.shape[0], 28,28)/225
x_test = x_test.reshape(x_test.shape[0], 28,28)/225

# one hot encoding
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

print(y_train.shape)                                    # (60000, 10)

#2. model

# gridsearch에 넣기위한 모델(모델에 대한 명시 : 함수로 만듦)
def build_model(drop=0.5, optimizer = 'adam'):
    inputs = Input(shape= (28,28 ), name = 'input')
    x = LSTM(512, activation = 'relu', name = 'hidden1', return_sequences=False)(inputs)
    x = Dense(256, activation = 'relu', name = 'hidden2')(x)
    x = Dense(128, activation = 'relu', name = 'hidden3')(x)
    outputs = Dense(10, activation = 'softmax', name = 'output')(x)
    model = Model(inputs = inputs, outputs = outputs)
    model.compile(optimizer = optimizer, metrics = ['acc'],
                  loss = 'categorical_crossentropy')
    return model

# parameter
def create_hyperparameters():
    batches = [10, 20, 30, 40, 50]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = np.linspace(0.1, 0.5, 5)                           # 
    return{'batch_size' : batches, 'optimizer': optimizers, 
           'drop': dropout}                                       # dictionary형태

# wrapper
from keras.wrappers.scikit_learn import KerasClassifier # sklearn에서 쓸수 있도로 keras모델 wrapping
model = KerasClassifier(build_fn = build_model, verbose = 1)

hyperparameters = create_hyperparameters()

# gridsearch
from sklearn.model_selection import GridSearchCV,  RandomizedSearchCV
from sklearn.metrics import accuracy_score
search = RandomizedSearchCV(model, hyperparameters, cv = 3)            # cv = cross_validation
# with n_jobs=1 it uses 100% of the cpu of one of the cores. Each process is run in a different core.
# # 모형 최적화 병렬/분산 처리¶
# 모형 최적화를 위해서는 많은 반복 처리과 계산량이 필요하다. 보통은 복수의 프로세스, 혹은 컴퓨터에서 여러가지 다른 하이퍼 모수를 가진 모형을 훈련시킴으로써 모형 최적화에 드는 시간을 줄일 수 있다.

# Scikit-Learn 패키지의 기본 병렬 처리¶
# GridSearchCV 명령에는 n_jobs 라는 인수가 있다. 디폴트 값은 1인데 이 값을 증가시키면 내부적으로 멀티 프로세스를 사용하여 그리드서치를 수행한다. 만약 CPU 코어의 수가 충분하다면 n_jobs를 늘릴 수록 속도가 증가한다.

# fit
search.fit(x_train, y_train)
acc = search.score(x_test, y_test, verbose = 0)


# print('최적의 매개변수 : ', model.best_estimator_)
# y_pred = model.predict(x_test)
# print("최종 정답률 = ", accuracy_score(y_test, y_pred) )

print(search.best_params_)   # serch.best_estimator_
print("acc : ", acc)