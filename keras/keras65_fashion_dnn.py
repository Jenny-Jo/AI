# sequential로 완성
# 하단에 주석으로 acc와 loss 결과 명시하시오

from keras.datasets import fashion_mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense


(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print(x_train.shape) #(60000, 28, 28)
print(x_test.shape)  #(10000, 28, 28)
print(y_train.shape) #(60000,)
print(y_test.shape)  #(10000,)

from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train.shape) #(60000, 10)

x_train = x_train.reshape(60000,28*28).astype('float32')/255
x_test = x_test.reshape(10000,28*28).astype('float32')/255

print('x_train.shape: ', x_train.shape)
print('x_test.shape : ' , x_test.shape)

model = Sequential()

model.add(Dense(100, activation='relu', input_shape =(28*28, )))
model.add(Dense(300, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(400, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()


model.compile (loss = 'categorical_crossentropy', optimizer= 'adam', metrics= ['acc'])
model.fit(x_train, y_train, epochs = 150, batch_size= 256, verbose=1,  validation_split = 0.2)

loss, acc = model.evaluate(x_test, y_test, batch_size=256)


print('loss: ', loss)
print('acc:', acc)

#loss:  0.8904106648519635
# acc: 0.8920000195503235