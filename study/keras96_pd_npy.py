# 95번을 복붙, save한 넘파이 가져와서 모델 완성
import numpy as np

datasets = np.load('./data/csv.npy')

print(datasets)
print(datasets.shape)
# x = datasets

x = datasets[:, :4]
y = datasets[:, 4]

print(x)
print(y)

print(x.shape)
print(y.shape)

from keras.utils import np_utils
y = np_utils.to_categorical(y)

print(y)
print(y.shape)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8)

# 모델 구성

from keras.layers import Dense
from keras.models import Sequential


model = Sequential()

model.add(Dense(10, input_dim=4, activation='relu' ))
model.add(Dense(10, activation='relu' ))
model.add(Dense(10, activation='relu' ))
model.add(Dense(10, activation='relu' ))
model.add(Dense(10, activation='relu' ))
model.add(Dense(3, activation='softmax'))

# 컴파일

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'] )
model.fit(x_train, y_train, epochs=10, batch_size=100)


loss,acc = model.evaluate(x_test, y_test, batch_size=1)
print("loss : ", loss)
print("acc : ", acc)

