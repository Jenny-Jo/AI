#shape이 가장 중요!!

from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM

#1. Data
x = array([[1,2,3],[2,3,4], [3,4,5],[4,5,6]]) #(4, 3)

y = array([4,5,6,7]) #(4, )
y2 = array([[4,5,6,7]]) #(1,4) 
y3 = array([[4], [5], [6], [7]]) ##(4, 1) 1개짜리가 네개 있다

print("x.shape : ", x.shape)
print("y.shape : ", y.shape) ## .shape 중요!! ##
print("y2.shape : ", y2.shape) ## .shape 중요!! ##
print("y3.shape : ", y3.shape) ## .shape 중요!! ##



# x.shape :  (4, 3)
# y.shape :  (4,) scalar 4개 ## 이거 잘 틀림 ##



# x = x.reshape(4, 3, 1) #검산> 전체 다 곱해서 같으면 맞음
x = x.reshape(x.shape[0], x.shape[1], 1 ) #x.shape (4,1)의 0번째, 1번째 것 출력/ 나중에 코딩 고칠 때 용이함
print(x.shape) # (4, 3, 1) 4행 3열을 한개씩 작업하겠다 *****왜????? 3차원을 만들기 위하여
print(x)


#2. 모델구성
model = Sequential()
model.add(LSTM(10, activation = 'relu', input_shape= (3, 1))) #노드가 10개/ 여기서부터 dense 모델
#??????????????
model.add(Dense(5))
model.add(Dense(1))

model.summary()

'''
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
lstm_1 (LSTM)                (None, 10)                480 #왜 480인가? 
_________________________________________________________________
dense_1 (Dense)              (None, 5)                 55
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 6
=================================================================
Total params: 541
Trainable params: 541
Non-trainable params: 0
_________________________________________________________________
'''

#3. 실행
model.compile (optimizer='adam', loss = 'mse')
model.fit(x,y, epochs=100)

x_input = array([5, 6, 7]) #평가를 해보고 싶어서 스칼라 3개, 벡터 1개 짜리 넣어줘
x_input = x_input.reshape(1,3,1) #(3, ) //input_shape = (3,1)애 맞춰서 형태에 맞춰주기

#[[[5]
# [6]
# [7]]]

print(x_input)

yhat = model.predict(x_input)
print(yhat)
