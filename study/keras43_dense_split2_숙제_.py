#42번 복붙, Dense로 리뉴얼
# train, test 분리(8:2)
# 마지막 6개 행을 predict로 만들기
# vallidation (train 20%)
# 튜닝 해야함!!
# earlystopping


import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Input
from array import array

#1. Data
a = np.array(range(1,101))
size = 5                                #time_steps = 4 

def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1) : 
        subset = seq [ i: (i+size)]
        aaa.append(subset)
    return np.array(aaa)

dataset = split_x(a,size) 
print("dataset :",dataset)
print("dataset.shape:", dataset.shape)  #(96,5)
print(type(dataset))                    #<class 'numpy.ndarray'>

# x = dataset[0:6,0:4]
# y = dataset[0:6,4:5]
x = dataset[ : , 0:4]                   #[모든 행, 0,1,2,3열 ]
y = dataset[:, 4]                       #[모든 행, 인덱스 4]


from sklearn.model_selection import train_test_split
x1, x_predict, y1, y_predict = train_test_split ( 
    #x, y, random_state=66, shuffle=True,#default option, 원래대로 하고 싶으면 shuffle=false
    x, y, shuffle= False, train_size=90/96
    )

x_train, x_test, y_train, y_test = train_test_split ( 
    #x, y, random_state=66, shuffle=True,#default option, 원래대로 하고 싶으면 shuffle=false
    x, y, shuffle= False, train_size=0.8
    )


#2.모델구성

model = Sequential()

model.add(Dense(10, input_shape=(4, )))
model.add(Dense(5))
model.add(Dense(1))

model.summary()



#3.실행

from keras.callbacks import EarlyStopping
from keras.losses import mse
es = EarlyStopping(monitor='loss', patience=5, mode='auto')



model.compile (optimizer='adam', loss = 'mse', metrics = ['mse'])
model.fit(x_train, y_train, epochs=1, batch_size =1 , verbose =0,
         callbacks = [es], validation_split=0.2)

#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=10)

y_predict = model.predict(x_predict)
print('loss: ', loss)
print('mse:', mse)
print('y_predict: ', y_predict)

