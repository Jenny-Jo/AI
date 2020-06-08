from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Input,Dropout, Conv2D, Flatten
from keras.layers import Dense, MaxPooling2D
import numpy as np
# 1. data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape) # (60000, 28, 28)
print(x_test.shape)  # (10000, 28, 28)

# x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float')/255 
#                                                     # .astype('float')  굳이 안써도 됨
# x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float')/255 
                        # 4차원으로 변환            / 255개의 데이타를 나눠줘서 minmax로 바꿔주는 효과

x_train = x_train.reshape(x_train.shape[0], 28*28).astype('float')/255 
x_test = x_test.reshape(x_test.shape[0], 28*28).astype('float')/255 
 # 2차원으로 구성해 Dense모델 구성


# OneHotEncoding . keras에서 제공하는건 categorical임. 0에서부터 시작함, 차원 확인해야 함
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

print(y_train.shape) # (60000, 10)


# 2. model
def build_model(drop = 0.5, optimizer = 'adam'):
    inputs = Input(shape=(28*28, ), name ='input')
    x = Dense(512, activation='relu', name = 'hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(256, activation = 'relu', name ='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(128, activation = 'relu', name ='hidden2')(x)
    x = Dropout(drop)(x)
    outputs = Dense(10, activation='softmax')
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'])
    return model
# fit은 랜덤서치나 그리드서치에서 함

# 하이퍼 파라미터 구성

def create_hyperparameters():
    batches = [10,20,30,40,50]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = np.linspace(0.1, 0.5, 5) 
    return{"batch_size" : batches, "optimizer":optimizers, "drop" : dropout }
    # dictionary  형태가 하이퍼파라미터에 들어감
    
# wrapper class 땡겨온다/ keras에 scikit learn 싼다

from keras.wrappers.scikit_learn import KerasClassifier #KerasRegressor  분류, 회귀
model = KerasClassifier(build_fn= build_model, verbose = 1 )
# sklearn에서 쓸 수 있게 랩핑한거다

hyperparameters = create_hyperparameters()

# 3. 훈련
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
search = GridSearchCV(model, hyperparameters, cv=3) # cross validation = 3
search.fit(x_train, y_train)

print(search.best_params_)
