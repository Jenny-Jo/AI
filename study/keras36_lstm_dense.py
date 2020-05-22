#<keras35_lstm_sequences.py>

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

input1 = Input (shape = (3, 1))
input1 = LSTM (30)(input1)
input1 = Dense (30)(input1)
input1 = Dense (30)(input1)
input1 = Dense (30)(input1)
input1 = Dense (30)(input1)




# input1 = Input (shape= (3, 1))
# input2 = LSTM (30)(input1)
# input3 = Dense (30)(input2)
# input4 = Dense (30)(input3)
# input5 = Dense (30)(input4)
# input4 = Dense (30)(input5)
# input4 = Dense (30)(input4)

# output 변수 input 변수 맞추기

output1 = Dense(30)(input1)
output1 = Dense(30)(output1)
output1 = Dense(30)(output1)
output1 = Dense(30)(output1)
output1 = Dense(30)(output1)
output1 = Dense(1)(output1)

model = Model(inputs= input1, outputs = output1)

model.summary() 

#3. 실행
model.compile (optimizer='adam', loss = 'mse')

from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss', patience=5, mode='auto')
model.fit(x1, y, epochs=800, callbacks = [es])


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
