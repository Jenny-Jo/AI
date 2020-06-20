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
# << hyperparameter : 활성화 함수 >>
# 전결합층이 입력을 선형 변환한 것을 출력하나, 활성화 함수로  비선형을 갖게 하여 선형 분리 불가능한 데이터에 대응하기 위해 씀
# 1) sigmoid : 0, 1 안에 출력 (출력이 작아 학습속도 느려)
# 2) ReLU : Rectified Linear Unit / Relu(x) = 0 (x<0), 1 (x>=0)

model.add(Activation('sigmoid'))


# Network Structure
# 은닉층 많아지면 -> 입력층에 가까운 가중치 적절 갱신 어렵고/ 학습 진행 느려
# 은닉층 유닛수 많아지면 -> 중요성 낮은 특징량 추출 > 과학습하기 쉬워져
# 네트워크 구조는 경험에 근거하여 결정하는 경향

model = Sequential()
model.add(Dense(256, input_dim=784, activation='sigmoid'))
model.add(Dense(128, activation='sigmoid'))
model.add(Dropout(rate=0.5))
model.add(Dense(10, activation='softmax'))

def funcA():
    global epochs
    epochs =5

def funcB():
    global epochs
    epochs =10

def funcC():
    global epochs
    epochs =60
    
funcA()
# funcB()
# funcC()


# 컴파일
# heperparameter : lr : learning rate
sgd = optimizers.SGD(lr=0.1)
# hyperparameter : 최적화 함수 optimizer, 오차 함수 loss
# metrics는 평가 함수라 학습 자체에 관계가 없다
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['acc'])

# loss : 학습시 출력과 y값(지도 데이터)과의 차이를 평가 / 손실 함수 최소화하도록 오차역전파법으로 각 층의 가중치가 갱신
# 1) Squared Error - 회귀 - 최소치 부근에서 천천히 갱신 - 학습이 수렴하기 쉽다
# 2) cross-entropy error- 이항분류 - 예측라벨과 정답 라벨 값은 가까울 수록 작은 값이 됨/ 0~1 사이에 있는 두 숫자 차이 평가

# Optimizer : 오차 함수를 미분으로 구한 값을 학습속도, epoch 수, 과거 가중치 갱신량 근거로 가중치 갱신에 어떻게 반영할지 정하는 것

# 2. 학습
# hyperparameter 배치처리크기, epoch 수
history= model.fit(x_train, y_train, batch_size=500, epochs=5, verbose=1, validation_data=(x_test, y_test))
# fit의 출력인 acc
print('====================  ')


# 3. 모델 분류
score = model.evaluate(x_test, y_test, verbose=1)
#  일반화 정확도 = 모델에 테스트 데이터 전달 시 분류 정확도
print('evaluate loss:{0[0]}\nevaluate acc:{0[1]}'.format(score))
print(score)

# 테스트 데이터의 첫 10장을 표시한다
for i in range(10):
    plt.subplot(1, 10, i+1)
    plt.imshow(x_test[i].reshape(28,28),'gray')

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


