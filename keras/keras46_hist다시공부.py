
#40번
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM

#1. Data
a = np.array(range(1,101))
size = 5                    #time_steps = 4 

def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1) : 
        subset = seq [ i: (i+size)]
        aaa.append(subset)
    return np.array(aaa)

dataset = split_x(a,size) #(6, 5)
print(dataset)
print(dataset.shape)
print(type(dataset)) #<class 'numpy.ndarray'>

# x = dataset[0:6,0:4]
# y = dataset[0:6,4:5]
x = dataset[ : , 0:4] #[모든 행, 0,1,2,3열 ]
y = dataset[ : , 4]     #[모든 행, 인덱스 4]

print(x.shape)
print(y.shape)

x = np.reshape (x, (96,4,1))
# x = x.reshape(x.shape[0], x.shape[1], 1) 
print(x.shape)


#2.모델구성- 불러오기
from keras.models import load_model
model = load_model('./model/save_keras44.h5')

#충돌이 난다?
model.add(Dense(10, name = 'new1'))
model.add(Dense(10, name = 'new2'))
model.add(Dense(10, name = 'new3'))
model.add(Dense(1, name = 'new4')) # 땡겨쓴거 외에도 나머지는 튜닝해야 함


model.summary()

# sequential model에서 model 추가할 때?

#3.실행
from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss', patience=5, mode='auto')

model.compile (optimizer='adam', loss = 'mse', metrics= ['acc'])###

#4.훈련
#################  hist  ######################################

hist = model.fit(x, y, epochs=50, batch_size =1 , verbose =1, validation_split=0.2,
         callbacks = [es] )

print(hist) #<keras.callbacks.callbacks.History object at 0x000001CC67ADF788>
print("hist.keys : ", hist.history.keys) ###dict_keys(['loss', 'mse'])

import matplotlib.pyplot as plt

# plt.plot(x,y) or plt.plot(y)
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])

plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])

# plt.plot(hist.history['val_loss'])
plt.title('loss & acc')
plt.ylabel('loss, acc')
plt.xlabel('epoch')
plt.legend(['train loss','test loss', 'train acc', 'test acc'])
plt.show()
###################################################################
'''
#4. 평가, 예측
loss, acc = model.evaluate(x, y)

y_predict = model.predict(x)
print('loss: ', loss)
print('mse:', mse)
print('y_predict: ', y_predict)
'''
