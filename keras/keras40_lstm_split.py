# 튜닝 해야함!!
# keras40_lstm_split1.py
# earlystopping
# LSTM 모델을 완성하시오

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Input

#1. Data
a = np.array(range(1,11))
size = 5                    #time_steps = 4 ?

def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1) : 
        subset = seq [ i: (i+size)]
        aaa.append(subset)
    return np.array(aaa)

dataset = split_x(a,size) #(6, 5)
print(dataset)
# [[ 1  2  3  4  5] 
#  [ 2  3  4  5  6] 
#  [ 3  4  5  6  7] 
#  [ 4  5  6  7  8] 
#  [ 5  6  7  8  9] 
#  [ 6  7  8  9 10]]
(6, 5)
print(dataset.shape)
print(type(dataset)) #<class 'numpy.ndarray'>

# x = dataset[0:6,0:4]
# y = dataset[0:6,4:5]
x = dataset[ : , 0:4] #[모든 행, 0,1,2,3열 ]
y = dataset[:, 4]     #[모든 행, 인덱스 4]

print(x.shape) #(6, 4)
print(y.shape) #(6, 4, 1) 

x = np.reshape (x, (6,4,1))
# x = x.reshape(x.shape[0], x.shape[1], 1) 
print(x.shape) #(6, 4, 1)
print(x)
'''
[[[1]
  [2]
  [3]
  [4]]

 [[2]
  [3]
  [4]
  [5]]

 [[3]
  [4]
  [5]
  [6]]

 [[4]
  [5]
  [6]
  [7]]

 [[5]
  [6]
  [7]
  [8]]

 [[6]
  [7]
  [8]
  [9]]]
  '''

#2.모델구성
model = Sequential()
model.add(LSTM(100, input_shape=(4,1)))
model.add(Dense(200))
model.add(Dense(200))
model.add(Dense(1))

model.summary()



#3.실행
from keras.callbacks import EarlyStopping
from keras.losses import mse
es = EarlyStopping(monitor='loss', patience=5, mode='auto')

model.compile (optimizer='adam', loss = 'mse', metrics= ['mse'])
model.fit(x, y, epochs=800, batch_size =1 , verbose =1,
         callbacks = [es])

#4. 평가, 예측
loss, acc = model.evaluate(x, y)

y_predict = model.predict(x)
print('loss: ', loss)
print('mse:', mse)
print('y_predict: ', y_predict)

