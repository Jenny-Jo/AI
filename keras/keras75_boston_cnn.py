# (?, 13)
# 회귀

# PCA? fit하고 transform//sklearn    

import numpy as np
from sklearn.datasets import load_boston
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, Input
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

x_train = x_train.reshape(404, 2, 1, 1)
x_test = x_test.reshape(102, 2, 1, 1)
print(x_train.shape)
print(x_test.shape)

###
model = Sequential()

model.add(Conv2D(30, (1,1), input_shape=(2,1,1), padding='same',activation='relu'))
model.add(Conv2D(30, (1,1), padding ='same', activation='relu'))
# model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(1, activation='relu'))

model.summary()

###
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
earlystopping = EarlyStopping(monitor='val_loss', patience=10, mode='auto', verbose=1)
tensorboard = TensorBoard(log_dir='graph', histogram_freq=0, write_graph=True, write_images=True)
modelpath='./model/{epoch:02d}-{val_loss:4f}.hdf5'
checkpoint = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')


model.compile (loss = 'mse', optimizer='adam', metrics=['mse'])
hist = model.fit(x_train, y_train, epochs=10, batch_size=100, verbose =2, callbacks=[earlystopping, tensorboard, checkpoint])

model.summary()


###

loss_mse = model.evaluate(x_test, y_test, batch_size=64)
loss = hist.history['loss']
val_loss =  hist.history['val_loss']
mse = hist.history['mse']
val_mse = hist.history['val_mse']

print('mse : ', loss_mse[0])
print('val_mse:', val_mse)
print('loss_mse:', loss_mse)


 