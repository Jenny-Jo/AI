
# 1. 데이타 구성
from keras.datasets import cifar100
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt


(x_train, y_train), (x_test, y_test) = cifar100.load_data()



print(x_train.shape) #(50000, 32, 32, 3)
print(x_test.shape)  #(10000, 32, 32, 3)
print(y_train.shape) #(50000, 1)
print(y_test.shape)  #(10000, 1)

from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train.shape) #(50000,100)

x_train = x_train.reshape(50000,96,32).astype('float32')/255
x_test = x_test.reshape(10000,96,32).astype('float32')/255


model = Sequential()

model.add(LSTM(100, input_shape=(96,32))) #(784,1) 보단 빠름 /너무 한쪽으로만 치우치면 느림
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100)) # output shape

model.summary()

#3. 훈련                     
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics= ['acc']) 

from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
es = EarlyStopping(monitor = 'val_loss', patience = 10, mode = 'auto', verbose = 1)
tb_hist = TensorBoard(log_dir='graph', histogram_freq=0, write_graph=True, write_images=True)
modelpath='./model/{epoch:02d}-{val_loss:4f}.hdf5'
checkpoint = ModelCheckpoint(filepath = modelpath, monitor='val_loss', 
                            save_best_only=True, mode='auto')
plt.show()                            


hist = model.fit(x_train, y_train, epochs= 100, batch_size= 1024, verbose = 2,
                 validation_split=0.2,
                 callbacks = [es, checkpoint, tb_hist] )

#4. 평가
loss_acc = model.evaluate(x_test, y_test, batch_size= 1024)

loss = hist.history['loss']
val_loss = hist.history['val_loss']
acc = hist.history['acc']
val_acc = hist.history['val_acc']

print('loss:', loss_acc[0])
print('acc:', loss_acc[1])
print('val_acc: ', val_acc)
print('loss_acc:' , loss_acc)

###

import matplotlib.pyplot as plt

plt.figure(figsize= (10, 6))

plt.subplot(2, 1, 1)
plt.plot(hist.history['loss'], marker='.', c='red', label ='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('loss')
plt.legend(loc='upper right')

plt.subplot(2, 1, 2)
plt.plot(hist.history['acc'], marker='.', c='red', label ='acc')
plt.plot(hist.history['val_acc'], marker='.', c='blue', label='val_loss')
plt.grid()
plt.title('accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.xlabel('epoch')
plt.legend(['acc','val_acc'])

plt.show()