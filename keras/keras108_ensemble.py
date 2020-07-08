# github 참조
# 1. data
import numpy as np
x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y1_train = np.array([1,2,3,4,5,6,7,8,9,10])
y2_train = np.array([1,0,1,0,1,0,1,0,1,0])

# 2. modeling
from keras.models import Sequential,Model
from keras.layers import Dense, Input

input1 = Input(shape=(1,))
x1 = Dense(100)(input1)
x1 = Dense(100)(x1)
x1 = Dense(100)(x1)

x2 = Dense(50)(x1)
output1 = Dense(1)(x2)

x3 = Dense(70)(x1)
x3 = Dense(70)(x3)
output2 = Dense(1, activation='sigmoid')(x3)

model = Model(inputs = input1, outputs=[output1, output2])

model.summary()

# complie, train
model.compile(loss =['mse','binary_crossentropy'],optimizer='adam', metrics=['mse','acc'])
model.fit([x_train],[y1_train, y2_train], epochs=100, batch_size=1)

# 4. 평가 예측
loss = model.evaluate(x_train, [y1_train, y2_train])
print('loss : ', loss)

x1_pred = np.array([11,12,13,14])
y_pred = model.predict(x1_pred)
print(y_pred)
