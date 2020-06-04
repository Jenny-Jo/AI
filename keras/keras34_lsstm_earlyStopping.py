# <keras34_lstm_earlyStopping.py
#shape이 가장 중요!!
#<keras29_lstm3_scale.py>에서 함수형으로 바꿈
#predict 50,60,70 맞추기

from numpy import array
from keras.models import Model # sequential 에서 Model로 바꿈
from keras.layers import Dense, LSTM, Input

#1. Data
x1 = array([[1,2,3],[2,3,4], [3,4,5],[4,5,6],
          [5,6,7],[6,7,8],[7,8,9],[8,9,10],
          [9,10,11],[11,12,13],
          [20,30,40],[30,40,50],[40,50,60]]) 


y = array([4,5,6,7,8,9,10,11,12,13,50,60,70]) 

x1_predict = array([50,60,70]) #(3, )

print("x1.shape : ", x1.shape) #(13, 3)
print("y.shape : ", y.shape) #  (13, )# .shape 중요!! ##

x1 = x1.reshape(x1.shape[0], x1.shape[1], 1 ) 

print(x1.shape)
print(x1)


'''
              행         열         몇 개씩 자르는지
x의 shape = (batch_size, timesteps, feature)
batch size =행의 크기대로 자른다
feature은 열을 자른다

input_shape = (timestpes/열, feature)
              (input_length,input_dim)

'''
#2. 모델구성
# from keras.models import Sequential, Model
# from keras.layers import Dense, Input

input1 = Input(shape=(3,1))  ###(3, )이 아니라

dense1 = LSTM(100)(input1) #### 이번 줄 추가?????? 모델 추가와 dense 추가의 차이점은?
dense1 = Dense(100)(dense1)
dense1 = Dense(100)(dense1)
dense1 = Dense(100)(dense1)

output1= Dense(1)(dense1) #??????

model = Model(inputs = input1, outputs = output1)
model.summary()


#3. 실행
from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss', patience=5, mode='auto')

model.compile (optimizer='adam', loss = 'mse')
model.fit(x1, y, epochs=300, callbacks = [es])


#4. 평가
x1_predict = x1_predict.reshape(1,3,1) #(3, ) //input_shape = (3,1)애 맞춰서 형태에 맞춰주기

#변수명만 변경해줬음

# x_input = array([5, 6, 7]) #평가를 해보고 싶어서 스칼라 3개, 벡터 1개 짜리 넣어줘
# x_input = x_input.reshape(1,3,1) #(3, ) //input_shape = (3,1)애 맞춰서 형태에 맞춰주기

#[[[5]
# [6]
# [7]]]

print(x1_predict)
y_predict = model.predict(x1_predict)
print(y_predict)

#결과가 들쑥날쑥해서 잘하고 있는지 모르겠음
