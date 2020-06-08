# activation, layer, epochs, adam, batchsize
# .best_estimator_ : 최고 점수를 낸 파라미터를 가진 모형
# .best_params_ : 최고점수를 낸 파라미터
# .best_score_ : 최고 점수
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Input, Dropout, Conv2D, Flatten, MaxPooling2D, Dense
import numpy as np

#1. data
(x_train, y_train),(x_test, y_test) = mnist.load_data()

print(x_train.shape)                                   # (60000, 28, 28)
print(x_test.shape)                                    # (10000, 28, 28)

x_train = x_train.reshape(x_train.shape[0], 28*28)/225
x_test = x_test.reshape(x_test.shape[0], 28*28)/225

# one hot encoding
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

print(y_train.shape)                                    # (60000, 10)

#2. model

# gridsearch에 넣기위한 모델(모델에 대한 명시 : 함수로 만듦)
def build_model(drop=0.5, optimizer = 'adam'):
    inputs = Input(shape= (28*28, ), name = 'input')
    x = Dense(512, activation = 'relu', name = 'hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(256, activation = 'relu', name = 'hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(128, activation = 'relu', name = 'hidden3')(x)
    x = Dropout(drop)(x)
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
search = GridSearchCV(model, hyperparameters, cv = 3)            # cv = cross_validation

# fit
search.fit(x_train, y_train)

print(search.best_params_)   # serch.best_estimator_

'''
파라미터는 모델 내부에서 결정되는 변수입니다. 또한 그 값은 데이터로부터 결정됩니다.
 무슨 말인지 예를 들어 설명해보겠습니다. 
 한 클래스에 속해 있는 학생들의 키에 대한 정규분포를 그린다고 합시다. 
 규분포를 그리면 평균(μ)과 표준편차(σ) 값이 구해집니다. 
 여기서 평균과 표준편차는 파라미터(parameter)입니다. 
 파라미터는 데이터를 통해 구해지며 (They are estimated or learned from data), 모델 내부적으로 결정되는 값입니다. 
 사용자에 의해 조정되지 않습니다. (They are often not set manually by the practitioner)

 하이퍼 파라미터는 모델링할 때 사용자가 직접 세팅해주는 값을 뜻합니다. 
 (They are often specified by the practitioner) learning rate나 서포트 벡터 머신에서의 C, sigma 값, KNN에서의 K값 등등 굉장히 많습니다. 
 머신러닝 모델을 쓸 때 사용자가 직접 세팅해야 하는 값은 상당히 많습니다. 그 모든 게 다 하이퍼 파라미터입니다. 
 하지만, 많은 사람들이 그런 값들을 조정할 때 그냥 '모델의 파라미터를 조정한다'라는 표현을 씁니다. 
 원칙적으로는 '모델의 하이퍼 파라미터를 조정한다'라고 해야 합니다.

하이퍼 파라미터는 정해진 최적의 값이 없습니다. 휴리스틱한 방법이나 경험 법칙(rules of thumb)에 의해 결정하는 경우가 많습니다.
 (They can often be set using heuristics) 베이지안 옵티미제이션과 같이 자동으로 하이퍼 파라미터를 선택해주는 라이브러리도 있긴 합니다.

 파라미터와 하이퍼 파라미터를 구분하는 기준은 사용자가 직접 설정하느냐 아니냐입니다. 
 사용자가 직접 설정하면 하이퍼 파라미터, 모델 혹은 데이터에 의해 결정되면 파라미터입니다.
 '''