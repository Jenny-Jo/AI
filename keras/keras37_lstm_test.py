# 실습 : LSTM layer 5개 이상 엮어서 Dense 결과를 이겨내시오! 85 이상 나오게 해야.
# 
from numpy import array
from keras.models import Model              # sequential 에서 Model로 바꿈
from keras.layers import Dense, LSTM, Input

#1. Data
x1 = array([[1,2,3],[2,3,4], [3,4,5],[4,5,6],
          [5,6,7],[6,7,8],[7,8,9],[8,9,10],
          [9,10,11],[11,12,13],
          [20,30,40],[30,40,50],[40,50,60]]) 


y = array([4,5,6,7,8,9,10,11,12,13,50,60,70]) 

x1_predict = array([55,65,75])              # (3, )

print("x1.shape : ", x1.shape)              # (13, 3)
print("y.shape : ", y.shape)                # (13, )# .shape 중요!! ##

x1 = x1.reshape(x1.shape[0], x1.shape[1], 1 ) #(13, 3, 1)

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
from keras.models import Sequential, Model
from keras.layers import Dense
model = Sequential()

############################################################################

model.add(LSTM (500, input_length=3, input_dim=1, return_sequences=True)) 
# (13,3,1) 행무시 > input_dim = feature 1 > (none, 3, 1)
# Dense는 2차원을 받아들이고, LSTM은 3차원 받아들임 / return_squences는 2차원을 3차원으로 변환

model.add(LSTM(600, return_sequences= True))
model.add(LSTM(500, return_sequences= True))
model.add(LSTM(500, return_sequences= True))
model.add(LSTM(500, return_sequences= False))

############################################################################

model.add(Dense(500))
model.add(Dense(300))
model.add(Dense(1))


model.summary()

'''
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
lstm_1 (LSTM)                (None, 3, 10)             480

# 1개 input dim 1에서 output node 10개 (이게 output shape에서 feture됨. (None, 3, 10)), 480개로 증폭됨

_________________________________________________________________
lstm_2 (LSTM)                (None, 10)                840

# 840 = 4* (10 +1 bias + 10)*10
# output node의 갯수는 그 다음 시작의 feature 이다.

_________________________________________________________________
dense_1 (Dense)              (None, 10)                110
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 11
=================================================================
Total params: 1,441
Trainable params: 1,441
Non-trainable params: 0
_________________________________________________________________


'''



##################################################################################

#3. 실행
model.compile (optimizer='adam', loss = 'mse')

from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss', patience=5, mode='auto')
model.fit(x1, y, epochs=700, callbacks = [es])


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
