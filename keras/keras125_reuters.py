from keras.datasets import reuters
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. data
(x_train, y_train),(x_test, y_test) = reuters.load_data(num_words = 1000, test_split=0.2)

print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)
# (8982,) (2246,)
# (8982,) (2246,)


print(x_train[0])
print(y_train[0])
# [1, 2, 2, 8, 43, 10, 447, 5, 25, 207, 270, 5, 2, 111, 16, 369, 186, 90, 67, 7, 89, 5, 19,
#  102, 6, 19, 124, 15, 90, 67, 84, 22, 482, 26, 7, 48, 4, 49, 8, 864, 39, 209, 154, 6, 151, 6, 83, 11, 15, 22, 155, 
# 11, 15, 7, 48, 9, 2, 2, 504, 6, 258, 6, 272, 11, 15, 22, 134, 44, 11, 15, 16, 8, 197, 2, 90, 67, 52, 29, 209, 30, 32, 132, 6, 109, 15, 17, 12]
# 3

# print(x_train[0].shape) 한줄이라 안나옴
print(len(x_train[0])) # 87개
# 일정하지 않다 >>> padding 빈자리를 0으로 채우기

category = np.max(y_train) + 1
print('category:', category)
# category: 46

y_distribution = np.unique(y_train)
print(y_distribution)
# [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
#  24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45]

y_train_pd = pd.DataFrame(y_train)
bbb = y_train_pd.groupby(0)[0].count()
print('bbb',bbb)
print(bbb.shape)
# groupby 숙지하기

# bbb 0
# 0       55
# 1      432
# 2       74
# 3     3159
# 4     1949
# 5       17
# 6       48
# 7       16
# 8      139
# 9      101
# 10     124
# 11     390
# 12      49
# 13     172
# 14      26
# 15      20
# 16     444
# 17      39
# 18      66
# 19     549
# 20     269
# 21     100
# 22      15
# 23      41
# 24      62
# 25      92
# 26      24
# 27      15
# 28      48
# 29      19
# 30      45
# 31      39
# 32      32
# 33      11
# 34      50
# 35      10
# 36      49
# 37      19
# 38      19
# 39      24
# 40      36
# 41      30
# 42      13
# 43      21
# 44      12
# 45      18
# Name: 0, dtype: int64
# (46,)

from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

x_train = pad_sequences(x_train, maxlen=100, padding='pre')
x_test = pad_sequences(x_test, maxlen=100, padding='pre')


print(len(x_train[0]))
print(len(x_train[-1]))
# 87
# 105

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(x_train.shape, x_test.shape)
# (8982, 100) (2246,)

# 2. modeling
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Flatten
from keras.metrics import accuracy

model = Sequential()
model.add(Embedding(1000, 100, input_length=100)) # 맨위에서 1000으로 잡아줌/output node/??몰라여
model.add(LSTM(60))
model.add(Dense(46, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics =['acc'])
history = model.fit(x_train, y_train, batch_size = 100, epochs =10, validation_split=0.2)
acc = model.evaluate(x_test, y_test)[1]
print('acc:', acc)

# 시각화
y_val_loss = history.history['val_loss']
y_loss = history.history['loss']

plt.plot(y_val_loss, marker='.', c='red', label='TestSet Loss')
plt.plot(y_loss, marker='.', c='blue', label='TestSet Loss')
plt.legend(loc = 'upper right')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()


