# loss at compile
# loss optimizer
# 'adam' 90프로는 경사하강법으로 씀

# 1. data

import numpy as np
x = np.array([1,2,3,4])
y = np.array([1,2,3,4])

# 2. modeling
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(10, input_dim = 1, activation='relu'))
model.add(Dense(3))
model.add(Dense(100))
model.add(Dense(1))

# hyper parameter가 2개나 늘었고, the most used hyperparameter is learning rate
from keras.optimizers import Adam, RMSprop, SGD, Adadelta, Adagrad, Nadam
optimizer = Adam(lr = 0.001)              # 0.013134053908288479, 3.45393
# optimizer = RMSprop(lr = 0.001)          #0.0013843015767633915, 3.4504988 이게 제일 좋은 듯
# optimizer = SGD(lr = 0.001)             # 0.08092157542705536,  3.3419425
# optimizer = Adadelta(lr = 0.001)        # 6.919477939605713 ,  0.13778175
# optimizer = Adagrad(lr = 0.001)        #  0.2491438090801239, 2.834703
# optimizer = Nadam(lr = 0.001)          #  0.3181155323982239,  3.1656237


model.compile(loss='mse', optimizer=optimizer ,metrics=['mse'])

model.fit(x, y, epochs=100)

loss = model.evaluate(x, y)
print('loss:', loss)


pred1 = model.predict([3.5])

print(pred1)

