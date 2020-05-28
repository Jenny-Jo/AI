from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt


(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape) 
print(x_test.shape) 
print(y_train.shape) 
print(y_test.shape) 


from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train.shape) 

x_train = x_train.reshape(50000,3072).astype('float32')/255
x_test = x_test.reshape(10000,3072).astype('float32')/255

print('x_train.shape: ', x_train.shape)
print('x_test.shape : ' , x_test.shape)

model = Sequential()

model.add(Dense(100, activation='relu', input_shape =(3072, )))
model.add(Dense(300, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(400, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(10, activation='softmax')) ## softmax 꼭 써야해!!

model.summary()

model.compile (loss = 'categorical_crossentropy', optimizer= 'adam', metrics= ['acc'])
model.fit(x_train, y_train, epochs = 150, batch_size= 256, verbose=1,  validation_split = 0.2)

loss, acc = model.evaluate(x_test, y_test, batch_size=256)


print('loss: ', loss)
print('acc:', acc)

