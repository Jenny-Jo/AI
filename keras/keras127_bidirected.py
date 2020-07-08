1# 0 or 1 로
from keras.datasets import imdb, reuters
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. data
(x_train, y_train),(x_test, y_test) = imdb.load_data(num_words = 1000000)# ,test_split=0.2)

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

from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

x_train = pad_sequences(x_train, maxlen=100, padding='pre')
x_test = pad_sequences(x_test, maxlen=100, padding='pre')


print(len(x_train[0]))
print(len(x_train[-1]))
# 87
# 105

# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

print(x_train.shape, x_test.shape)
# (8982, 100) (2246,)

# 2. modeling
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Flatten, Conv1D, Dropout, Activation, Bidirectional # 연산 두번 하는거다?
from keras.metrics import accuracy
from keras.layers.pooling import MaxPooling1D

model = Sequential()
# model.add(Embedding(1000, 100, input_length=100)) # 맨위에서 1000으로 잡아줌/output node/??몰라여
model.add(Embedding(2000, 100)) # 맨위에서 1000으로 잡아줌/output node/??몰라여

model.add(Conv1D(10, 5, padding='valid', activation='relu', strides=1))
model.add(MaxPooling1D(pool_size=4))

# model.add(Flatten())
model.add(Bidirectional(LSTM(60)))
model.add(Dense(1, activation='sigmoid'))


model.summary()
'''
model.compile(loss='binary_crossentropy', optimizer='adam', metrics =['acc'])
history = model.fit(x_train, y_train, batch_size = 100, epochs =10, validation_split=0.2)
acc = model.evaluate(x_test, y_test)[1]
print('acc:', acc)

# 시각화
y_val_loss = history.history['val_loss']
y_loss = history.history['loss']

plt.plot(y_val_loss, marker='.', c='red', label='TestSet val Loss')
plt.plot(y_loss, marker='.', c='blue', label='TestSet Loss')
plt.legend(loc = 'upper right')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()


# 1. imdb 검색해서 데이터내용 확인
# 2. word_size 전체데이터 부분 변경해서 최상값 확인
# 3. 주간과제 : groupby 사용법 하.........
# 4. 인덱스를 단어로 바꿔주는 함수 찾기 decode_review()
''''''word_index = imdb.get_word_index()

word_index = {k : (v+3) for k,v in word_index.items()}
word_index['<PAD>'] = 0
word_index['<START>'] = 1
word_index['<UNK>'] = 2
word_index['<UNUSED>'] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])''''''
# 5. 125, 126번 튠'''