# iris를 파이프라인 구성
# 당연히 RandomizedSearchCV 구성
# keras98 참조해서 만들기


from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from keras.utils import np_utils
from keras.layers import Input, Dropout, Conv2D, Flatten, MaxPooling2D, Dense
from keras.models import Sequential, Model
from keras.layers import Input, Dropout, Conv2D, Flatten, MaxPooling2D, Dense
import numpy as np


#1. data
iris = load_iris()

x = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, 
                                                    shuffle = True, random_state = 43)

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)


#2. model

# gridsearch에 넣기위한 모델(모델에 대한 명시 : 함수로 만듦)
def build_model( optimizer = 'adam'):
    drop=0.5
    inputs = Input(shape= (4, ), name = 'input')
    x = Dense(512, activation = 'relu', name = 'hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(256, activation = 'relu', name = 'hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(128, activation = 'relu', name = 'hidden3')(x)
    x = Dropout(drop)(x)
    outputs = Dense(3, activation = 'softmax', name = 'output')(x)
    model = Model(inputs = inputs, outputs = outputs)
    model.compile(optimizer = optimizer, metrics = ['acc'],
                  loss = 'categorical_crossentropy')
    return model

# wrapper
from keras.wrappers.scikit_learn import KerasClassifier # ML/sklearn에서 쓸수 있도로 keras모델 wrapping
model = KerasClassifier(build_fn = build_model, verbose = 1)

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
pipe = Pipeline([("scaler", MinMaxScaler()), ('model', model)])  
#                               모델, 전처리  
                                            # 여기 있는 모델명을 명시해줘야 함##
# pipe = make_pipeline(MinMaxScaler(), model)
                                    # 여기랑 같아야

# parameter
# def create_hyperparameters():
#     batches = [125,256,512]
#     optimizers = ['rmsprop','adam','adadelta']                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        op', 'adam', 'adadelta']
#     dropout = np.linspace(0.1, 0.5, 5)                           # 
#     return{'model__batch_size' : batches, 'model__optimizer': optimizers} 
#         #    'drop': dropout}                                       # dictionary형태

def create_hyperparameters(): # epochs, node, acivation 추가 가능
    batches = [128, 256, 512]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    # dropout = np.linspace(0.1, 0.5, 5)            
    activation = ['relu', 'elu']               
    return {'model__batch_size' : batches, 'model__optimizer': optimizers} #'model__act':activation}
        #    'model__drop': dropout}      

hyperparameters = create_hyperparameters()

search = RandomizedSearchCV(pipe, hyperparameters , cv = 5)

#3. fit
search.fit(x_train, y_train)


#4. evaluate, predict
acc = search.score(x_test, y_test)

print('최적의 매개변수 = ', search.best_params_)
print('acc : ', acc) # 할 때 마다 accuracy 표시