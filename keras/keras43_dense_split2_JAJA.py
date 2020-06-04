"""실습 1. train, test 분리할 것.                 (90행) 8 : 2 비율
   실습 2. 마지막 6개의 행을 predict로 만들고 싶다.
   실습 3. validatoion 을 넣을 것                 (train의 20%)
"""

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM

#1. 데이터
a = np.array(range(1,101))
size = 5                                         


# LSTM 모델을 완성하시오.
def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1):       # len = length  : 길이  i in range(6)  : [0, 1, 2, 3, 4, 5]
        subset = seq[i : (i + size)]           # i =0,  subset = a[ 0 : 5 ] = [ 1, 2, 3, 4, 5]
        aaa.append([item for item in subset])  # aaa = [[1, 2, 3, 4, 5]]
        # aaa.append(subset)
    print(type(aaa))
    return np.array(aaa)

dataset = split_x(a, size)
print(dataset)                                 
print(dataset.shape)                           # (96, 5)
print(type(dataset))                           # numpy.ndarray


'''
## train, test, predict 값 나누기 : slicing이용
# x, y 나누기
x = dataset[:90, 0:4]                            # [ : ] 모든행 가져오고, [0 : 4] 0~3까지
y = dataset[:90, 4]                              # [ : ] 모든행 가져오고, [  : 4] 4번째
print(x.shape)                          
print(y.shape)
# x_predict 값
x_predict = dataset[-6:, 0:4]
# y_predict = dataset[-6:, 4]                    # model로 에측 할 것이기 때문에 필요 없다.
print(x_predict.shape)
# train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8)
'''
## train, test, predict 값 나누기 : train_test_split이용
# x, y 나누기
x = dataset[:, 0:4]
y = dataset[:, 4]

# x_predict 값
from sklearn.model_selection import train_test_split
x1, x_predict, y1, y_predict = train_test_split(x, y, train_size = 90/96)

# train_test_split
x_train, x_test, y_train, y_test = train_test_split(x1, y1, train_size = 0.8)

#-----------------------------------------------------------------------------
# 두 개의 성능은 동일하다.
# 다만, slicing은 원론적으로 알 수 있고, 
#      train_test_split는 percentage(%)로 나눌 수 있어서 좀 더 편리하다.


# reshape( , , )
# x = x.reshape(90, 4, 1)
# x_predict = x_predict.reshape(6, 4, 1)
print(x_train.shape)
print(x_test.shape)
print(x_predict.shape)


#==================================================================================================
#2. 모델

model = Sequential()
model.add(Dense(100, input_shape= (4, )))                # input_length : time_step (열)
model.add(Dense(100))   
model.add(Dense(100))   
model.add(Dense(80))   
model.add(Dense(50))   
model.add(Dense(10))   
model.add(Dense(1))

# EarlyStopping
from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor = 'loss', patience=100, mode = 'min')


#3. 실행
model.compile(loss = 'mse', optimizer='adam', metrics= ['mse'])
model.fit(x_train, y_train, epochs =1, batch_size = 16 , validation_split= 0.25,
         callbacks = [es])                



#4. 평가, 예측
loss, mse = model.evaluate(x_test, y_test, batch_size= 16)

print('loss :',loss )
print('mse :',mse )


y_predict = model.predict(x_predict)
print(y_predict)