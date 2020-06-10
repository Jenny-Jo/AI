import numpy as np
import pandas as pd
from keras.layers import LSTM, Dense, Dropout, Conv1D,Flatten, MaxPooling1D, Input
from keras.models import Sequential, Model
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import warnings

warnings.filterwarnings('ignore')

# 1. data

# 1) 불러 오기
test_features = pd.read_csv("./data/dacon/comp3/test_features.csv",  header = 0, index_col = 0)
train_features = pd.read_csv("./data/dacon/comp3/train_features.csv", header = 0, index_col = 0)
train_target = pd.read_csv("./data/dacon/comp3/train_target.csv", header = 0, index_col = 0)

# 2) shape
print(train_features)   # (1050000, 5)
print(train_target)     # (2800, 4)
print(test_features)    # (262500, 5)

#  행 같게 맞춰줘야

# 3) x, y 나눠주기
x1 = train_features.values.reshape(2800, 375*5 )
y1 = train_target               # (2800, 4)
x_predict = test_features.values
# (x1이랑 열이 같게/인풋쉐입 같게)


# 4) Standard scaler, PCA
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x1)
x1 = scaler.transform(x1)
x_predict = scaler.transform(x1)

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(x1)
x1 = pca.transform(x1) 
x_predict = pca.transform(x_predict)





x_train, x_test, y_train, y_test = train_test_split(x1, y1, train_size =0.8,random_state=13)

def build_model(drop=0.5, optimizer = 'adam'):
    inputs = Input(shape= (2, ), name = 'input')
    x = Dense(512, activation = 'relu', name = 'hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(256, activation = 'relu', name = 'hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(128, activation = 'relu', name = 'hidden3')(x)
    x = Dropout(drop)(x)
    outputs = Dense(4, activation = 'softmax', name = 'output')(x)
    model = Model(inputs = inputs, outputs = outputs)
    model.compile(optimizer = optimizer, metrics = ['mse'],
                  loss = 'mse')
    return model

# parameter
def create_hyperparameters():
    batches = [10, 20, 30, 40, 50]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    # dropout = np.linspace(0.1, 0.5, 5)                           # 
    return{'batch_size' : batches, 'optimizer': optimizers
           }                                       # dictionary형태

# wrapper
from keras.wrappers.scikit_learn import KerasClassifier # sklearn에서 쓸수 있도로 keras모델 wrapping
model = KerasClassifier(build_fn = build_model, verbose = 1)

hyperparameters = create_hyperparameters()

# gridsearch
from sklearn.model_selection import GridSearchCV,  RandomizedSearchCV
from sklearn.metrics import accuracy_score
search = RandomizedSearchCV(model, hyperparameters, cv = 3, n_jobs = -1 )            # cv = cross_validation
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

y_predict = search.predict(x_predict) #

y_pred = pd.DataFrame({
    'id' : np.array(range(2800, 3500)),
    'X'  : y_predict[:,0],
    'Y'  : y_predict[:,1],
    'M'  : y_predict[:,2],
    'V'  : y_predict[:,3]
})
print(y_pred)

y_pred.to_csv('./dacon/comp3/comp3_y_pred.csv',index=False)




import numpy as np
from sklearn.metrics import r2_score
from docutils.io import Input
from bokeh.model import Model

def kaeri_metric(y_true, y_pred):
    '''
    y_true: dataframe with true values of X,Y,M,V
    y_pred: dataframe with pred values of X,Y,M,V
    
    return: KAERI metric
    '''
    
    return 0.5 * E1(y_true, y_pred) + 0.5 * E2(y_true, y_pred)


### E1과 E2는 아래에 정의됨 ###

def E1(y_true, y_pred):
    '''
    y_true: dataframe with true values of X,Y,M,V
    y_pred: dataframe with pred values of X,Y,M,V
    
    return: distance error normalized with 2e+04
    '''
    
    _t, _p = np.array(y_true)[:,:2], np.array(y_pred)[:,:2]
    
    return np.mean(np.sum(np.square(_t - _p), axis = 1) / 2e+04)


def E2(y_true, y_pred):
    '''
    y_true: dataframe with true values of X,Y,M,V
    y_pred: dataframe with pred values of X,Y,M,V
    
    return: sum of mass and velocity's mean squared percentage error
    '''
    
    _t, _p = np.array(y_true)[:,2:], np.array(y_pred)[:,2:]
    
    
    return np.mean(np.sum(np.square((_t - _p) / (_t + 1e-06)), axis = 1))

y_pred1 = model.predict(x_test)
print(kaeri_metric(y_test, y_pred1))
print(E1(y_test, y_pred1))
print(E2(y_test, y_pred1))

print((kaeri_metric(y_test, y_pred1) + E1(y_test, y_pred1) +E2(y_test, y_pred1))/3)




