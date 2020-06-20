import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.layers import Activation, Dense, Dropout
from keras.models import Sequential, load_model
from keras import optimizers
from keras.utils.np_utils import to_categorical
from keras.utils.vis_utils import plot_model
# %matlotlib inline

# Data set
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape)
print(y_train.shape)
x_train = x_train.reshape(x_train.shape[0],784)[:6000]
x_test = x_test.reshape(x_test.shape[0], 784)[:1000]
y_train = to_categorical(y_train)[:6000]
y_test = to_categorical(y_test)[:1000]

# 1. Model 생성
model = Sequential()
model.add(Dense(256, input_dim = 784))
model.add(Activation('sigmoid'))
model.add(Dense(10))
model.add(Dense(10))

model.add(Dense(128))
model.add(Activation('sigmoid'))
model.add(Dropout(rate=0.5))
model.add(Dense(10))
model.add(Activation('softmax'))

model.summary()

# 컴파일
sgd = optimizers.SGD(lr=0.1)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['acc'])

# 2. 학습
history= model.fit(x_train, y_train, batch_size=500, epochs=5, verbose=1, validation_data=(x_test, y_test))
# fit의 출력인 acc
print('====================  ')

# 3. 학습
score = model.evaluate(x_test, y_test, verbose=1)
#  일반화 정확도 = 모델에 테스트 데이터 전달 시 분류 정확도
print('evaluate loss:{0[0]}\nevaluate acc:{0[1]}'.format(score))
print(score)

# 테스트 데이터의 첫 10장을 표시한다
for i in range(10):
    plt.subplot(1, 10, i+1)
    plt.imshow(x_test[i].reshape(28,28),'gray')

# 4. 모델 분류
# x_test의 첫 10장의 예측된 라벨 표시
pred = np.argmax(model.predict(x_test[0:10]), axis=1)
print(pred)
# [7 2 1 0 4 1 7 7 6 7]

plt.plot(history.history['acc'], label='acc', ls='-', marker='o')
plt.plot(history.history['val_acc'], label='val_acc', ls='-', marker='x')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(loc='best')
plt.show()

# plot_model(model, 'model125.png', show_layer_names=False)
# image = plt.imread('model125.png')
# plt.figure(dpi=150)
# plt.imshow(image)
# plt.show()
# OSError: `pydot` failed to call GraphViz.Please install GraphViz (https://www.graphviz.org/) and ensure that its executables are in the $PATH.
'''
Train on 6000 samples, validate on 1000 samples
Epoch 1/5
2020-06-19 19:33:19.569487: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library cublas64_100.dll
6000/6000 [==============================] - 0s 66us/step - loss: 2.4074 - acc: 0.1432 - val_loss: 2.0204 - val_acc: 0.5370
Epoch 2/5
6000/6000 [==============================] - 0s 8us/step - loss: 2.0618 - acc: 0.2687 - val_loss: 1.8109 - val_acc: 0.6580
Epoch 3/5
6000/6000 [==============================] - 0s 8us/step - loss: 1.8382 - acc: 0.3940 - val_loss: 1.6257 - val_acc: 0.6990
Epoch 4/5
6000/6000 [==============================] - 0s 8us/step - loss: 1.6646 - acc: 0.4815 - val_loss: 1.4654 - val_acc: 0.7320
Epoch 5/5
'''


