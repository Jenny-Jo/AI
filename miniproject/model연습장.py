# from numpy import array
# from keras.models import Model, Sequential              # sequential 에서 Model로 바꿈
# from keras.layers import Dense, LSTM, Input, Embedding

# examples = [
#   '뭐 ***인가 *** *** ***들',
#   '웃기넼ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ', 
#   '힘내세요! 응원합니다', 
#   '못생김ㅋㅋㅋ 한심', 
#   '너무 멋져요!'
# ]

# model = Sequential()
# model.add(Embedding(max_words, 8, input_length=maxlen))
# model.add(LSTM(32))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))

# from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, Dropout
from keras.preprocessing import sequence
from keras.utils import np_utils

import numpy
import tensorflow as tf
import matplotlib.pyplot as p0lt
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling1D
from keras.layers.core import Activation
import pandas as pd
from sklearn.model_selection import train_test_split
from keras import preprocessing
from matplotlib import pyplot as plt

# 불러오기
a = pd.read_csv('F:\\Study\\miniproject\\x_data\\1.csv', index_col=0, header =0, sep='\t')
b = pd.read_csv('F:\\Study\\miniproject\\x_data\\2.csv',index_col=0, header =0, sep='\t')


x_data = pd.concat([a, b], axis =0)
x = x_data.values[:,1] # (5915,)
print(x.shape)


y_data = pd.read_csv('F:\\Study\\miniproject\y_data\\욕취합.csv',index_col=0, header =0, sep='\t')
y = y_data.values.reshape(49*20, )
print(y.shape)         # (980,)


print(type(x))
print(type(y))

x_train, x_test = train_test_split(x,test_size = 0.8)
y_train, y_test = train_test_split(y, test_size = 0.8)

x_train = sequence.pad_sequences(x_train, maxlen=100)
x_test = sequence.pad_sequences(x_test, maxlen=100)
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

print(x_train.shape)
print( x_test.shape)
print(y_data.shape)

# (2456, 3)
# (3459, 3)
# (50, 21)
# (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=5000)

maxlen=20
x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)

model = Sequential()
model.add(Embedding(5000,100))
model.add(Dropout(0.5))
model.add(Conv1D(64,5, padding= 'valid', activation='relu', strides=1))
model.add(MaxPooling1D(pool_size=4))
model.add(LSTM(55))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(x_train, y_train, batch_size=100, epochs=5, validation_data=(x_test,y_test))

#테스트 정확도 출력
print("\n 정확도 : %.4f" % (model.evaluate(x_test, y_test)[1]))

#테스트셋의 오차
y_vloss = history.history['val_loss']

#학습셋의 오차
y_loss = history.history['loss']

x_len = numpy.arange(len(y_loss))
plt.plot(x_len, y_vloss, marker='.', c="red", label="Testset_loss")
plt.plot(x_len, y_loss, marker='.', c="blue", label="Trainset_loss")

#그래프테 그리드를 주고 레이블을 표시
plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
