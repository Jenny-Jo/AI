from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tqdm import keras


import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist        # MNIST에 이미 저장되어있는 데이타셋을 불러온다




# 1. Data=======================================================
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train[0])


print('y_train : ' , y_train[0])
'''
#print(x_train[0]) 결과
Downloading data from https://s3.amazonaws.com/img-datasets/mnist.npz
11493376/11490434 [==============================] - 162s 14us/step
    0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
    0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
    0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
    0   0   0   0   0   0   0   0   0   0]
    0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   3  18  18  18 126 136
  175  26 166 255 247 127   0   0   0   0]
 [  0   0   0   0   0   0   0   0  30  36  94 154 170 253 253 253 253 253
  225 172 253 242 195  64   0   0   0   0]
 [  0   0   0   0   0   0   0  49 238 253 253 253 253 253 253 253 253 251
   93  82  82  56  39   0   0   0   0   0]
 [  0   0   0   0   0   0   0  18 219 253 253 253 253 253 198 182 247 241
    0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0  80 156 107 253 253 205  11   0  43 154
    0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0  14   1 154 253  90   0   0   0   0
    0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0 139 253 190   2   0   0   0
    0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0  11 190 253  70   0   0   0
    0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0  35 241 225 160 108   1
    0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0  81 240 253 253 119
   25   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   0  45 186 253 253
  150  27   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0  16  93 252
  253 187   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0 249
  253 249  64   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   0  46 130 183 253
  253 207   2   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0  39 148 229 253 253 253
  250 182   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0  24 114 221 253 253 253 253 201
   78   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0  23  66 213 253 253 253 253 198  81   2
    0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0  18 171 219 253 253 253 253 195  80   9   0   0
    0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0  55 172 226 253 253 253 253 244 133  11   0   0   0   0
    0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0 136 253 253 253 212 135 132  16   0   0   0   0   0   0
    0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
    0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
    0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
    0   0   0   0   0   0   0   0   0   0]]
    '''
# y_train :  5

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

x_train = x_train.reshape(60000, 28, 28, 1). astype('float32')/255
x_test = x_test.reshape(10000, 28, 28, 1). astype('float32')/255

# CNN model에 넣기 위해 4차원으로/ 동일한 가로세로/x 픽셀 보면 255까지 정수형태로 나옴 /완전 진한 검정색/ 
# min max는 0부터 1, 실수형이라 float로 함
# 0부터 1까지니까 255로 나눔 / 픽셀의 Max 값이 255인 걸 (0~255) 이미 알고 있고, Max 값 알고 싶으면 MinMax값 나오게 하는 함수 쓰면 된다??????

# x_predict ---------------------------------------
from sklearn.model_selection import train_test_split
x_train, x_predict, y_train, y_predict = train_test_split(
    x_train, y_train, shuffle=False, train_size = 0.8
)



#2. 모델구성 ==========================================

model = Sequential()

model.add(Conv2D(50, (2,2), input_shape = (28,28, 1)))
model.add(Conv2D(50,(5,5)))
model.add(Conv2D(50, (5,5)))
model.add(Conv2D(50, (2,2)))
model.add(Conv2D(50, (2,2)))
model.add(Conv2D(50, (2,2)))

model.add(MaxPooling2D(pool_size=2))
model.add(Flatten())
model.add(Dense(10, activation = 'softmax'))

model.summary()

# 3. 실행 ===================================
from keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor= 'loss', patience= 10, mode = 'auto')

model.compile (loss = 'categorical_crossentropy', optimizer= 'adam', metrics= ['accuracy'])
model.fit(x_train, y_train, epochs = 10, batch_size= 256, verbose=0, callbacks= [early_stopping], validation_split = 0.2)

#4. 평가, 예측 ========================================
loss, acc = model.evaluate(x_test, y_test, batch_size=256)


print('loss: ', loss)
print('acc:', acc)

#0.982 accuracy 올리기