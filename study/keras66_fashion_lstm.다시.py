# sequential로 완성
# 하단에 주석으로 acc와 loss 결과 명시하시오

from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape) #(50000, 32, 32, 3)
print(x_test.shape)  #(10000, 32, 32, 3)
print(y_train.shape) #(50000, 1)  
print(y_test.shape)  #(10000, 1) 

from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train.shape) #(50000, 10)


x_train = x_train.reshape(50000,32,96).astype('float32')/255
x_test = x_test.reshape(10000,32,96).astype('float32')/255

print('x_train.shape: ', x_train.shape)
print('x_test.shape : ' , x_test.shape)

model = Sequential()

model.add(LSTM(100, input_shape=(32,96))) #(784,1) 보단 빠름 /너무 한쪽으로만 치우치면 느림
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(10))


# EarlyStopping
from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor = 'val_loss', patience = 50, mode = 'auto', verbose = 1)

#3. 훈련                     
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics= ['acc']) 
model.fit(x_train, y_train, epochs= 100, batch_size= 1024, verbose = 2,
                 validation_split=0.2,
                 callbacks = [es] )

#4. 평가
loss, acc = model.evaluate(x_test, y_test, batch_size= 64)
print('loss: ', loss)
print('acc: ', acc)