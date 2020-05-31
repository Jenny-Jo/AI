# x shape은 (?, 13)

# 1. data
import numpy as np
from sklearn.datasets import load_boston
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt

dataset = load_boston()
x = dataset.data
y = dataset.target

print(x.shape, y.shape) #(506, 13) (506, )

###
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(x)
x_scaled = scaler.transform(x)
print(x_scaled)

from sklearn.decomposition import PCA

pca = PCA(n_components= 2)
pca.fit(x_scaled)
x_pca = pca.transform(x_scaled)
print(x_pca)

###
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x_pca, y, random_state=66, shuffle= True, train_size = 0.8
)

print(x_train.shape) #(404, 2)
print(x_test.shape)  #(102, 2)
print(y_train.shape) #(404, )
print(y_test.shape)  #(102, )

x_train = x_train.reshape(404, 2, 1)
x_test = x_test.reshape(102, 2, 1)
print(x_train.shape)
print(x_test.shape)


# 2. 모델
model  = Sequential()

model.add(LSTM (100, input_shape = (2, 1)))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(1))

#1) early stopping
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
earlystopping = EarlyStopping(monitor = 'mse', patience = 50, mode= 'auto', verbose=1)

#2)tensorboard
tensorboard = TensorBoard(log_dir= 'graph', histogram_freq=0, write_graph = True, write_images=True)
#3)modelcheckpoint
modelpath = './model/{epoch:02d}-{val_loss:4f}.hdf5'
checkpoint = ModelCheckpoint(filepath = modelpath, monitor='val_loss', save_best_only=True, mode='auto')

# 3.훈련
model.compile(loss = 'mse', optimizer='adam', metrics=['mse'])
hist = model.fit (x_train, y_train, epochs=10, batch_size = 100, verbose= 2, callbacks = [earlystopping, tensorboard, checkpoint])

model.summary()
# 4. 평가

loss_mse = model.evaluate(x_test, y_test, batch_size=64)

loss = hist.history['loss']
val_loss = hist.history['val_loss']
mse = hist.history['mse']
val_mse = hist.history['val_mse']       

print('mse : ', loss_mse[0])
print('val_mse: ', val_mse)
print('loss_mse: ' , loss_mse)


## 4) plt 
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
plt.plot(hist.history['mse'], marker='.', c='red', label='mse')
plt.plot(hist.history['val_mse'], marker='.', c='blue', label='val_mse')
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
