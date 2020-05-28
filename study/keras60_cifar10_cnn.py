
# 1. 데이타 구성
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D
from keras.layers import Flatten, MaxPooling2D, Dropout
import matplotlib.pyplot as plt


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print("len(x_train) : ", len(x_train))
print("x_train : ", x_train)



print("x_train[0] : ", x_train[0])
print('y_train[0] : ', y_train[0]) # [6]

print(x_train.shape) #(50000, 32, 32, 3)
print(x_test.shape) #(10000, 32, 32, 3)
print(y_train.shape) #(50000, 1)
print(y_test.shape) #(10000, 1)

# plt.imshow(x_train[0])
# plt.show()

from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train.shape) #(50000,1)

x_train = x_train.reshape(50000,32,32,3).astype('float32')/255
x_test = x_test.reshape(10000,32,32,3).astype('float32')/255


# 2. 모델구성

from keras.models import Model
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Input
from keras.layers import Dropout

input1 = Input(shape = (32, 32, 3))

dense1 = Conv2D(200, (3, 3), padding = 'same')(input1)
dense1 = Conv2D(200, (3, 3), padding = 'same')(dense1)
dense1 = Conv2D(200, (3, 3), padding = 'same')(dense1)

maxpool1 = MaxPooling2D(pool_size=2)(dense1)
drop1 = Dropout(0.2)(maxpool1)

flat = Flatten()(drop1)
output1 = Dense( 10 , activation='softmax')(flat)

model = Model(inputs = input1, outputs = output1)
model.summary()

#3.실행
from keras.callbacks import EarlyStopping
from keras.layers.core import Dropout
early_stopping = EarlyStopping(monitor = 'loss', patience=10, mode = 'auto' )

model.compile (loss = 'categorical_crossentropy', optimizer='adam', metrics= ['acc'])
model.fit(x_train, y_train, epochs = 100, batch_size=256,
verbose= 1, callbacks = [early_stopping], validation_split= 0.2)

#4.평가,예측
loss, acc = model.evaluate(x_test, y_test, batch_size=256)

print('loss: ', loss)
print('acc : ', acc)

