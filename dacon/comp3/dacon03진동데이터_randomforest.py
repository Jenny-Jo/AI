import numpy as np
import pandas as pd
from keras.layers import LSTM, Dense, Dropout, Conv1D,Flatten, MaxPooling1D
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor

# 1. data

# 1) 불러 오기
test_features = pd.read_csv("./data/dacon/comp3/test_features.csv",  header = 0, index_col = 0)
train_features = pd.read_csv("./data/dacon/comp3/train_features.csv", header = 0, index_col = 0)
train_target = pd.read_csv("./data/dacon/comp3/train_target.csv", header = 0, index_col = 0)

# 2) shape
print(train_features)   # (1050000, 5)
print(train_target)     # (2800, 4)
print(test_features)    # (262500, 5)

def preprocessing_KAERI(data) :
    '''
    data: train_features.csv or test_features.csv
    
    return: Random Forest 모델 입력용 데이터
    '''
    
    # 충돌체 별로 0.000116 초 까지의 가속도 데이터만 활용해보기 
    _data = data.groupby('id').head(30)
    
    # string 형태로 변환
    _data['Time'] = _data['Time'].astype('str')
    
    # Random Forest 모델에 입력 할 수 있는 1차원 형태로 가속도 데이터 변환
    _data = _data.pivot_table(index = 'id', columns = 'Time', values = ['S1', 'S2', 'S3', 'S4'])
    
    # column 명 변환
    _data.columns = ['_'.join(col) for col in _data.columns.values]
    
    return _data
train_features = preprocessing_KAERI(train_features)
test_features = preprocessing_KAERI(test_features)
print(f'train_features {train_features.shape}')
print(f'test_features {test_features.shape}')

# train_features (2800, 120)
# test_features (700, 120)

# 3) x, y 나눠주기
x1 = train_features.values
y1 = train_target               # (2800, 4)
x_predict = test_features.values


x_train, x_test, y_train, y_test = train_test_split(x1, y1, train_size =0.8)

# 2. model
model = RandomForestRegressor(n_jobs=-1, random_state=0)


# model = Sequential()
# model.add(Conv1D(100,3, input_shape = (375, 5), padding = 'same', activation='relu'))
# model.add(Dropout(0.3))
# model.add(Conv1D(100,3, padding = 'same', activation='relu'))
# model.add(Dropout(0.3))
# model.add(MaxPooling1D())
# model.add(Flatten())
# model.add(Dense(10, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(10, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(10, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(10, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(10, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(10, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(10, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(10, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(10, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(10, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(10, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(10, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(4, activation='relu'))

# model.summary()

# 3. compile, fit
# model.compile(optimizer='adam', loss='mse', metrics=['mse'])
model.fit(x_train,y_train)

# 4. evaluation, predict
# loss, mse = model.evaluate(x_test,y_test)
y_predict = model.predict(x_predict)
# print('loss,mse : ', loss, mse)
print(y_predict.shape)

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




