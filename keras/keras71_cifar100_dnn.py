# sequential로 완성
# 하단에 주석으로 acc와 loss 결과 명시하시오
from keras.datasets import cifar100
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = cifar100.load_data()

print(x_train.shape) # (50000, 32, 32, 3)
print(x_test.shape)  # (10000, 32, 32, 3)
print(y_train.shape) # (50000, 1)
print(y_test.shape)  # (10000, 1)

from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train.shape) # (50000, 100)

x_train = x_train.reshape(50000,32*32*3).astype('float32')/255
x_test = x_test.reshape(10000,32*32*3).astype('float32')/255

print('x_train.shape: ', x_train.shape)
print('x_test.shape : ', x_test.shape)

model = Sequential()

model.add(Dense(100, activation='relu', input_shape =(32*32*3, )))
model.add(Dense(300, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(400, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='softmax'))

model.summary()

model.compile (loss = 'categorical_crossentropy', optimizer= 'adam', metrics= ['acc'])
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

tb_hist = TensorBoard(log_dir = 'graph', histogram_freq = 0, write_graph = True, write_images= True)
early_stopping = EarlyStopping(monitor= 'loss', patience=10, mode='auto')


plt.show()

modelpath='./model/sample/cifar100/{epoch:02d}-{val_loss:.4f}_checkpoint_best.hdf5'
checkpoint = ModelCheckpoint(filepath=modelpath, monitor= 'val_loss', verbose=1, save_best_only=True, save_weights_only=False)

hist = model.fit(x_train, y_train, epochs = 1, batch_size= 256, verbose=1,  validation_split = 0.2)

model.save('./model/sample/cifar100/cifar100_model_save.h5')
model.save_weights('./model/sample/cifar100/cifar100_save_weight1.h5')


loss_acc = model.evaluate(x_test, y_test, batch_size=256)

loss = hist.history['loss']
val_loss = hist.history['val_loss']
acc = hist.history['acc']
val_acc = hist.history['val_acc']

print('loss:', loss_acc[0])
print('acc:', loss_acc[1])
print('val_acc:', val_acc)
print('loss_acc:', loss_acc)

import matplotlib.pyplot as plt

plt.figure(figsize= (10, 6))

plt.subplot(2, 1, 1)
plt.plot(hist.history['loss'], marker='.', c = 'red', label = 'loss')
plt.plot(hist.history['val_loss'], marker = '.', c = 'blue', label = 'val_loss')
plt.grid()
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc = 'upper right')

plt.subplot(2, 1, 2)
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.grid()
plt.title('accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['acc', 'val_acc'])

plt.show()

