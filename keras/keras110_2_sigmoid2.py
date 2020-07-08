# github 참조
# 1. data
import numpy as np
x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,0,1,0,1,0,1,0,1,0])

# 2. modeling
from keras.models import Sequential,Model
from keras.layers import Dense, Input

model = Sequential()
model.add(Dense(10, input_shape = (1,)))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(1, activation='sigmoid'))
# sigmoid 0~1
#tanh -1 0

model.summary()

# complie, train
model.compile(loss =['binary_crossentropy'],optimizer='adam', metrics=['acc'])
model.fit(x_train,y_train, epochs=100, batch_size=1)

# 4. 평가 예측
loss = model.evaluate(x_train, y_train)
print('loss : ', loss)

x1_pred = np.array([11,12,13,14])
y_pred = model.predict(x1_pred)
print(y_pred)
