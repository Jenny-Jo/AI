# 200525 1000~
# LSTM 모델을 완성하시오.

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM

# 1. 데이터
a = np.array(range(1, 11))
size = 5

def split_x(seq, size):
    aaa = []        # 는 리스트
    for i in range(len(seq) - size + 1):
        subset = seq[i : (i+size)]
        aaa.append([item for item in subset])
    print(type(aaa))
    return np.array(aaa)

dataset = split_x(a, size)          # 6,5
print(dataset)
print(dataset.shape)                # 6,5
print(type(dataset))                # numpy.ndarray

x = dataset[:, 0:4]                 # : 모든 행, 그 다음 0:4
y = dataset[:, 4]                   # : 모든 행, 인덱스 4부분만 가져오겠다.

print(x.shape)                            # 6,4
print(y.shape)                            # 6,

x = np.reshape(x, (6,4,1))
print(x.shape)
# x = x.reshape(6, 4, 1) 같은 문법

'''
        x           y
[[[ 1  2  3  4 | 5]]

 [[ 2  3  4  5  6]]

 [[ 3  4  5  6  7]]

 [[ 4  5  6  7  8]]

 [[ 5  6  7  8  9]]

 [[ 6  7  8  9 10]]]

'''
'''
# = 이렇게 되는 것
x = np.array([[1,2,3,4],[2,3,4,5],[3,4,5,6],[4,5,6,7],[5,6,7,8],[6,7,8,9]])
y = np.array([5,6,7,8,9,10])
print("x_shape :", x.shape) # (6,4)
print("y_shape :", y.shape) # (6, )

x = x.reshape(x.shape[0], x.shape[1], 1) # (6,4,1)
print(x.shape)
'''

# 2. 모델구성
model = Sequential()
model.add(LSTM(300, input_shape=(4,1)))
model.add(Dense(200))
model.add(Dense(150))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(1))

model.summary()


# 3. 실행
model.compile(loss='mse', optimizer='adam', metrics = ['mse'] )

# 3-1. 얼리스타핑
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=100, mode='auto')

# 3-2. 훈련
model.fit(x, y, epochs=800, batch_size=32, verbose=2,
          callbacks=[early_stopping])
          # 여기서 batch_size 와 쉐이프의 batch_size 는 다르다.
          # 그 이유는 ? shape의 뱃치 사이즈는 총 행의 수,  fit에서의 뱃치 사이즈는 거기서 n개씩 작업하겠다는 것.

# 4. 예측
loss, mse = model.evaluate(x, y)
y_predict = model.predict(x)
print('loss:', loss)
print('mse:', mse)
print('y_predict:', y_predict)

# x_predict = x_predict.reshape(1,6,1)
# print(x_predict)

# # 4. 예측
# y_predict = model.predict(x_predict)
# print(y_predict)