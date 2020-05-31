import numpy as np

import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Dropout
from keras.layers import Flatten, MaxPooling2D, Input

from keras.datasets import cifar100
import matplotlib.pyplot as plt

#1. DATA
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
#colomn 백개 예제
#체크포인트, 시각화 텐서보드, cnn, dnn, lstm 

print(x_train[0])
print('y_train[0] : ', y_train[0])

print(x_train.shape) # (50000, 32, 32, 3)
print(x_test.shape)  # (10000, 32, 32, 3)
print(y_train.shape) # (50000, 1)
print(y_test.shape)  # (10000, 1)

x_train = x_train.reshape(50000,32,32,3).astype('float32')/255
x_test = x_test.reshape(10000,32,32,3).astype('float32')/255

from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train.shape) # (50000, 100)

#2.Model
##################################################
input1 = Input(shape = (32, 32, 3))

dense1 = Conv2D(200, (3, 3), padding = 'same')(input1)
dense1 = Conv2D(200, (3, 3), padding = 'same')(dense1)
dense1 = Conv2D(200, (3, 3), padding = 'same')(dense1)

maxpool1 = MaxPooling2D(pool_size=2)(dense1)
drop1 = Dropout(0.2)(maxpool1)

flat = Flatten()(drop1)
output1 = Dense( 100 , activation='softmax')(flat)

model = Model(inputs = input1, outputs = output1)
model.summary()

#3.실행
model.compile (loss = 'categorical_crossentropy', optimizer='adam', metrics= ['acc'])

from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.layers.core import Dropout

# 1)
tb_hist = TensorBoard(log_dir = 'graph', histogram_freq = 0, 
                      write_graph =True, write_images = True)
    # tensorboard --logdir=f:\study\git\graph
    
# 2)
early_stopping = EarlyStopping(monitor = 'loss', patience=10, mode = 'auto' )
# 3)
modelpath = './model/{epoch:02d} - {val_loss:.4f}.hdf5'                          # 파일 경로, 파일명 설정
checkpoint = ModelCheckpoint(filepath=modelpath, monitor='val_loss',
                             save_best_only= True, mode = 'auto')  
plt.show()

# 4)
hist = model.fit(x_train, y_train, epochs = 10, batch_size=256,
verbose= 1, callbacks = [early_stopping, checkpoint, tb_hist], validation_split= 0.2)


#4.평가,예측
loss_acc = model.evaluate(x_test, y_test, batch_size=256)

loss = hist.history['loss']
val_loss = hist.history['val_loss']
acc = hist.history['acc']
val_acc = hist.history['val_acc']

print('loss: ', loss_acc[0])
print('acc : ', loss_acc[1])
print('val_acc: ' , val_acc)
print('loss_acc: ' , loss_acc)

import matplotlib.pyplot as plt    

plt.figure(figsize = (10, 6))                     # 10 x 6인치의 판이 생김

# 1번 그림
plt.subplot(2, 1, 1)                              # (2, 1, 1) 2행 1열의 그림 1번째꺼 / subplot : 2장 그림               
plt.plot(hist.history['loss'], marker = '.', c = 'red', label = 'loss')                     
plt.plot(hist.history['val_loss'], marker = '.', c = 'blue', label = 'val_loss')                  
plt.grid()                                        # 격자 생성
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
# plt.legend(['loss','val_loss']) 
plt.legend(loc = 'upper right')                   # legend의 위치(location) 설정/ default = 제이 비어 있는 곳
                                                  # 위에 label이 지정되어서 안써도 된다.

# 2번 그림
plt.subplot(2, 1, 2)                              # (2, 1, 2) 2행 1열의 그림 2번째꺼               
plt.plot(hist.history['acc'])                     
plt.plot(hist.history['val_acc'])                  
plt.grid()                                        # 격자 생성
plt.title('accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['acc','val_acc'])

plt.show()                       
