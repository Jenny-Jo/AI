import numpy as np
from keras.datasets import mnist                          # keras에서 제공되는 예제 파일 
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Input
from array import array

mnist.load_data()                                         # mnist파일 불러오기

(x_train, y_train), (x_test, y_test) = mnist.load_data()  

print(x_train[0])                                         # 0 ~ 255까지의 숫자가 적혀짐 (color에 대한 수치)
print('y_train: ' , y_train[0])                           # 5

print(x_train.shape)                                      # (60000, 28, 28)
print(x_test.shape)                                       # (10000, 28, 28)
print(y_train.shape)                                      # (60000,)        : 10000개의 xcalar를 가진 vector(1차원)
print(y_test.shape)                                       # (10000,)



# 데이터 전처리 1. 원핫인코딩 : 당연하다             
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train.shape)                                      #  (60000, 10)

# 데티어 전처리 2. 정규화( MinMaxScalar )                                              
x_train = x_train.reshape(-1, 28,28).astype('float32') /255  
x_test = x_test.reshape(-1, 28, 28).astype('float32') /255.                                    


# 2. 모델구성
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Input #??
from keras.layers import Dropout                   

model = Sequential()

model.add(LSTM(100, input_shape=(28,28))) #(784,1) 보단 빠름
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(10)) # output shape


model.summary()

# EarlyStopping
from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor = 'val_loss', patience = 50, mode = 'auto', verbose = 1)

#3. 훈련                     
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics= ['acc']) 
model.fit(x_train, y_train, epochs= 1, batch_size= 64, verbose = 2,
                 validation_split=0.2,
                 callbacks = [es] )



#4. 평가
loss, acc = model.evaluate(x_test, y_test, batch_size= 64)
print('loss: ', loss)
print('acc: ', acc)

