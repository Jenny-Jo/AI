# 200525 1400~
# 40번 카피


import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM

# 1. 데이터
a = np.array(range(1, 101))
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

x = np.reshape(x, (96,4,1))
print(x.shape)
# x = x.reshape(6, 4, 1) 같은 문법


# 2. 모델
from keras.models import load_model
model = load_model('./model/save_keras44.h5')

model.add(Dense(11, name='dense_x'))
model.add(Dense(12, name='dense_x2'))       # 이게 전이 학습이다 ?
model.add(Dense(13, name='dense_x3'))
model.add(Dense(1, name='dense_x4'))

model.summary()


# 3. 실행
model.compile(loss='mse', optimizer='adam', metrics = ['acc'] )

# 3-1. 얼리스타핑
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=100, mode='auto')

# 3-2. 훈련
hist = model.fit(x, y, epochs=50, batch_size=1, verbose=2,
          validation_split=0.2,
          callbacks=[early_stopping])

print(hist)                # 자료형만 출력
print(hist.history.keys()) # dict_keys(['loss', 'mse']) 키 loss와 mse가 있는데 각각 벨류도 있을 것이다.

import matplotlib.pyplot as plt

plt.plot(hist.history['loss'])      # 히스토리에 저장된 애들을 끄집어온다.
plt.plot(hist.history['val_loss'])  # x,y값 둘다 드가거나 y값 집어넣거나
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('loss & acc')
plt.ylabel('loss, acc')
plt.xlabel('epoch')
plt.legend(['train loss', 'val loss', 'train acc', 'val acc'])
# 검증이 실질적으로 좋지 않게 나왔다. (val 그래프가 위로 치고 있으니)
plt.show()


'''
# 4. 예측
loss, mse = model.evaluate(x, y)
y_predict = model.predict(x)
print('loss:', loss)
print('mse:', mse)
print('y_predict:', y_predict)
'''