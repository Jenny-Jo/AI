# 과제 2 
# Sequential 형으로
# 하단에 주석으로 acc와 loss 결과 명시하시오

# Data
from keras.datasets import fashion_mnist
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D
from keras.layers import Flatten, MaxPooling2D, Dropout


(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
print(x_train[0])

print(x_train.shape) # (60000, 28, 28)
print(x_test.shape)  # (10000, 28, 28)
print(y_train.shape) # (60000, )
print(y_test.shape)  # (10000, )

from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train.shape) # (60000, 10)

x_train = x_train.reshape(60000,28,28).astype('float32')/255
x_test = x_test.reshape(10000,28,28).astype('float32')/255

print('x_train.shape: ', x_train.shape) #(60000, 784)
print('x_test.shape : ' , x_test.shape) #(10000, 784)


# 2. model 
from keras.models import Model
from keras.layers import Dense, Input,  Conv2D, Flatten, MaxPooling2D, Input
from keras.layers import Dropout
 

 ###?????????????????????????????????
model = Sequential()
model.add(Conv2D(10,(2,2), input_shape = (28,28)))
model.add(Conv2D(10, (3,3),padding = 'same' ))
model.add(Conv2D(100,(2,2), padding = 'same'))
model.add(MaxPooling2D(pool_size=2))
model.add(Flatten())
model.add(Dense(1))

model.summary()


# EarlyStopping
from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor = 'loss', patience=100, mode = 'min')


#3. 실행
model.compile(loss = 'mse', optimizer='adam', metrics= ['mse'])
model.fit(x_train, y_train, epochs =1, batch_size = 16 , validation_split= 0.25,
         callbacks = [es])  


#4. 평가, 예측
loss, mse = model.evaluate(x_test, y_test, batch_size= 16)

print('loss :',loss )
print('mse :',mse )

