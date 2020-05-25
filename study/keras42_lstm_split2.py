# 튜닝 해야함!!
# earlystopping
# LSTM 모델을 완성하시오

import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Input
from array import array

#1. Data
a = np.array(range(1,101))
size = 5                    #time_steps = 4 

def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1) : 
        subset = seq [ i: (i+size)]
        aaa.append(subset)
    return np.array(aaa)

dataset = split_x(a,size) #(96,5)
print(dataset)
print(dataset.shape)
print(type(dataset)) #<class 'numpy.ndarray'>

# x = dataset[0:6,0:4]
# y = dataset[0:6,4:5]
#####################################################
x = dataset[ : , 0:4] #[모든 행, 0,1,2,3열 ]
y = dataset[:, 4]     #[모든 행, 인덱스 4]
#####################################################

print(x)
print(y)

x = np.reshape (x, (96,4,1))
# x = x.reshape(x.shape[0], x.shape[1], 1) 
print(x.shape)


#2.모델구성
model = Sequential()

model.add(LSTM(10, input_shape=(4, 1)))
model.add(Dense(5))
model.add(Dense(1))



model.summary()



#3.실행
from keras.losses import mse
from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss', patience=5, mode='auto')



model.compile (optimizer='adam', loss = 'mse', metrics = ['mse'])
model.fit(x, y, epochs=1, batch_size =1 , verbose =1,
         callbacks = [es])

#4. 평가, 예측
loss, acc = model.evaluate(x, y)

y_predict = model.predict(x)
print('loss: ', loss)
print('mse:', mse)
print('y_predict: ', y_predict)

