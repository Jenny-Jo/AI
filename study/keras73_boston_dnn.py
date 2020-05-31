#회귀모델
import numpy as np
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt

# 1. Data

# data : x 값
# target : y값

dataset = load_boston()
x = dataset.data
y = dataset.target

# DNN으로 구현
print(x.shape)  # (506,13)
print(y.shape)  # (506, )

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(x)
x_scaled = scaler.transform(x)
print(x_scaled)

from sklearn.decomposition import PCA

pca = PCA(n_components=2) #차원 두개로 나눠준다
pca.fit(x_scaled)
x_pca = pca.transform(x_scaled)
print(x_pca)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_pca, y, random_state= 66, 
shuffle= True, train_size = 0.8)

print(x_train.shape) # (402, 2)
print(x_test.shape)  # (102, 2)
print(y_train.shape) #(404,)
print(y_test.shape) #(102,)


# 2. Model
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()

model.add(Dense(100, input_shape=(2, )))
model.add(Dense(200))
model.add(Dense(200))
model.add(Dense(200))
model.add(Dense(1))
model.summary()


# 3. 훈련

##### EarlyStopping & Modelcheckpoint & Tensorboard
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
es = EarlyStopping(monitor='loss', patience=50)

modelpath = './model/{epoch:02d}-{val_loss:.4f}.hdf5'
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss',
                     save_best_only=True, mode='auto')

tb = TensorBoard(log_dir='graph', histogram_freq=0,
                 write_graph=True, write_images=True)
     # (cmd에서) -> d: -> cd study -> cd graph -> tensorboard --logdir=.
     # 127.0.0.1:6006

# 3. 컴파일

model.compile(loss='mse', optimizer='adam', metrics=['mse'])
hist = model.fit(x_train, y_train,
          epochs=25, batch_size=64, verbose=2,
          validation_split=0.4,
          callbacks=[es, cp, tb])


##### 4. 평가, 예측
loss_mse = model.evaluate(x_test, y_test, batch_size=64)

loss = hist.history['loss']
val_loss = hist.history['val_loss']
mse = hist.history['mse']
val_mse = hist.history['val_mse']

print('mse 는 ', mse)
print('val_mse 는 ', val_mse)

# evaluate 종속 결과
print('loss, mse 는 ', loss_mse)


##### plt 시각화
import matplotlib.pyplot as plt

plt.figure(figsize=(10,7))

# 1
plt.subplot(2,1,1)
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')

# 2
plt.subplot(2,1,2)
plt.plot(hist.history['mse'])
plt.plot(hist.history['val_mse'])
plt.grid()
plt.title('mse')
plt.ylabel('mse')
plt.xlabel('epoch')
plt.legend(['mse', 'val_mse'])

plt.show()


# RMSE 구하기
y_predict = model.predict(x_test)
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))

# R2 구하기
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)