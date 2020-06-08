# 97번을 RandomizedSearchCV로 변경하시오
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Input, Dropout, Conv2D, Flatten, MaxPooling2D, Dense
import numpy as np



#1. data
(x_train, y_train),(x_test, y_test) = mnist.load_data()

print(x_train.shape)                                   # (60000, 28, 28)
print(x_test.shape)                                    # (10000, 28, 28)

x_train = x_train.reshape(x_train.shape[0], 28,28,1)/225
x_test = x_test.reshape(x_test.shape[0], 28,28,1)/225

# one hot encoding
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

print(y_train.shape)                                    # (60000, 10)

#2. model

# gridsearch에 넣기위한 모델(모델에 대한 명시 : 함수로 만듦)
def build_model(drop=0.5, optimizer = 'adam'):
    inputs = Input(shape= (28,28,1 ), name = 'input')
    x = Conv2D(100,(3,3),padding = 'same', activation = 'relu', name = 'hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Conv2D(100,(3,3),padding = 'same', activation = 'relu', name = 'hidden2')(x)
    x = Dropout(drop)(x)
    maxpool1 = MaxPooling2D(pool_size=2)(x)
    x = Flatten()(maxpool1)
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
search = RandomizedSearchCV(model, hyperparameters, cv = 3 )            # cv = cross_validation
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