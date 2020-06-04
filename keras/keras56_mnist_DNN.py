# x를 4차원에서 2차원으로 변형, Dense 모델에 넣어주기

from keras.models import Sequential
from keras.layers import Dense #####
import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist  

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train[0])


print('y_train : ' , y_train[0])


print(x_train.shape)                   #(60000, 28, 28)
print(x_test.shape)                    #(10000, 28, 28)
print(y_train.shape)                   #(60000,) 스칼라, 1 dim(vector)
print(y_test.shape)                    #(10000,)


print(x_train[0].shape) #(28,28) 짜리 
# plt.imshow(x_train[0], 'gray')
# plt.imshow(x_train[0])
# plt.show()



# 1. y-------------------------------------------------------------------
# Data 전처리 / 1. OneHotEncoding 큰 값만 불러온다 y
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train.shape) # (60000, 10) 6만장 10개로 증폭/ 아웃풋 dimension 10

# 0과 255 사이를 0과 1 사이로 바꿔줘

x_train = x_train / 255


# 2. x-------------------------------------------------------------------

# Data 전처리/ 2. 정규화 x
# 형을 실수형으로 변환
# # MinMax scaler (x - 최대)/ (최대 - 최소)

############ 4차원을 2차원으로#########
x_train = x_train.reshape(60000, 784).astype('float32')/255 ##??????
x_test  = x_test.reshape (10000, 784).astype('float32')/255 ##??????
######################################
print('x_train.shape: ', x_train.shape)
print('x_test.shape : ' , x_test.shape)




#2. 모델구성 ==========================================

model = Sequential()

model.add(Dense(100, activation='relu', input_shape =(784, )))
model.add(Dense(300, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(400, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(10, activation='softmax')) ## softmax 꼭 써야해!!

model.summary()

# 3. 실행 ===================================

model.compile (loss = 'categorical_crossentropy', optimizer= 'adam', metrics= ['acc'])
model.fit(x_train, y_train, epochs = 150, batch_size= 256, verbose=1,  validation_split = 0.2)

#4. 평가, 예측 ========================================
loss, acc = model.evaluate(x_test, y_test, batch_size=256)

print('loss: ', loss)
print('acc:', acc)


#99.25 
