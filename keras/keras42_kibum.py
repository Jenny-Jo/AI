# 200525 1200~
# LSTM 모델을 완성하시오.
# 실습 1. train, test 분리할 것 (8:2)
# 실습 2. 마지막 6개의 행을 predict 로 만들고 싶다. (90행을 test로 잡으면)
# 실습 3. validation 을 넣을 것 (train의 20%)

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM

# 1. 데이터
a = np.array(range(1, 101))
size = 5                    # timp_steps : 4

def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1):
        subset = seq[i : (i+size)]
        aaa.append([item for item in subset])
    print(type(aaa))
    return np.array(aaa)

dataset = split_x(a, size)          # 96,5
print(dataset)
print(dataset.shape)                # 96,5
print(type(dataset))                # numpy.ndarray

x = dataset[:90, 0:4]                 # : 모든 행, 그 다음 0:4
y = dataset[:90, 4]                   # : 모든 행, 인덱스 4부분만 가져오겠다.
x_pred = dataset[-6:, 0:4]

# 프레딕트
print(x.shape)                            # 90,4
print(x)                                  # 90,4
print(y.shape)                            # 90,
print(x_pred.shape)                       # 6,4
print(x_pred)                             # 6,4


x = np.reshape(x, (x.shape[0],x.shape[1],1))
print(x.shape)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state = 66, shuffle = True,
    # x, y, shuffle = False,
    train_size = 0.8)


# x = x.reshape(6, 4, 1) 같은 문법


# 2. 모델구성
model = Sequential()
model.add(LSTM(30, input_shape=(4,1)))
model.add(Dense(20))
model.add(Dense(15))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(1))

model.summary()



# 3. 실행
model.compile(loss='mse', optimizer='adam', metrics = ['mse'] )

# 3-1. 얼리스타핑
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=100, mode='auto')

# 3-2. 훈련
model.fit(x_train, y_train, epochs=800, batch_size=32, verbose=2, validation_split=0.25,
        #   validation_split=0.25, random_state = 66, shuffle=True,
          callbacks=[early_stopping])

x_pred = x_pred.reshape(6,4,1)
print(x_pred)

# 4. 평가, 예측
loss, mse = model.evaluate(x_test, y_test)
y_predict = model.predict(x_pred)
print('loss:', loss)
print('mse:', mse)
print('y_predict:', y_predict)


# RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))

# R2 구하기
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)
