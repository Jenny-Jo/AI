# 정확도 80% 이상
# epoch수 5 고정
# x_train, y_train, x_test, y_test 정의문 변경하지 말기

import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.layers import Activation, Dense, Dropout
from keras.models import Sequential, load_model
from keras import optimizers
from keras.utils.np_utils import to_categorical

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape)
print(y_train.shape)
x_train = x_train.reshape(x_train.shape[0],784)[:6000]
x_test = x_test.reshape(x_test.shape[0], 784)[:1000]
y_train = to_categorical(y_train)[:6000]
y_test = to_categorical(y_test)[:1000]


model = Sequential()
model.add(Dense(256, input_dim=784, activation='sigmoid'))
model.add(Dense(128, activation='sigmoid'))
model.add(Dropout(rate=0.5))
model.add(Dense(10, activation='softmax'))



sgd = optimizers.SGD(lr=0.1)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['acc'])

history= model.fit(x_train, y_train, batch_size=96, epochs=5, verbose=1, validation_data=(x_test, y_test))


score = model.evaluate(x_test, y_test, verbose=1)
print('evaluate loss:{0[0]}\nevaluate acc:{0[1]}'.format(score))
print(score)

