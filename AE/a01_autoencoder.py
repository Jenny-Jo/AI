# x를 4차원에서 2차원으로 변형, Dense 모델에 넣어주기
# keras 56_mnist_DNN.py 복붙


from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
import numpy as np
import matplotlib.pyplot as plt

#Datasets 불러오기
from tensorflow.keras.datasets import mnist  

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape)                   #(60000, 28, 28)
print(x_test.shape)                    #(10000, 28, 28)
print(y_train.shape)                   #(60000,) 스칼라, 1 dim(vector)
print(y_test.shape)                    #(10000,)

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

# 함수형

input_img = Input(shape= (784, ))
encoded = Dense(64, activation='relu')(input_img) # 784개 중 특성 32개 추출
decoded = Dense(784, activation='sigmoid')(encoded)

autoencoder = Model(input_img, decoded)

autoencoder.summary()
autoencoder.compile(optimizer='adam', loss= 'binary_crossentropy')
# autoencoder.compile(optimizer='adam', loss= 'mse')

autoencoder.fit(x_train, x_train, epochs=50, batch_size=256, validation_split=0.2) # y값이 x값이 됨

decoded_imgs = autoencoder.predict(x_test)

import matplotlib.pyplot as plt
n = 10
plt.figure(figsize=(20,4))
for i in range(n):
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    ax = plt.subplot(2, n, i+1+n)
    plt.imshow(decoded_imgs[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()
