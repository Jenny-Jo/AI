-#<keras31>가져옴
#shape이 가장 중요!!
#predict 50,60,70 맞추기

from numpy import array
from keras.models import Model
from keras.layers import Dense, LSTM, Input #여기를 잘 바꾸자

#1. Data
x1 = array([[1,2,3],[2,3,4], [3,4,5],[4,5,6],
          [5,6,7],[6,7,8],[7,8,9],[8,9,10],
          [9,10,11],[11,12,13],
          [20,30,40],[30,40,50],[40,50,60]]) 

x2 = array([[10,20,30],[20,30,40], [30,40,50],[40,50,60],
          [50,60,70],[60,70,80],[70,80,90,],[80,90,100],
          [90,100,110],[100,110,120],
          [2,3,4],[3,4,5],[4,5,6]]) #(13,3)

y = array([4,5,6,7,8,9,10,11,12,13,50,60,70]) 

x1_predict = array([55,65,75])
x2_predict = array([65,75,85])
#predict는 x1, x2 둘 다 있어야 함.

print("x1.shape : ", x1.shape)
print("x2.shape : ", x2.shape)

print("y.shape : ", y.shape) ## .shape 중요!! ##

x1 = x1.reshape(x1.shape[0], x1.shape[1], 1 ) 
x2 = x2.reshape(x2.shape[0], x2.shape[1], 1 ) 

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

input1 = Input(shape=(3, 1 )) #input에 shape만 들어가면 된다 (3, )이 아닌 이유??

dense1 = LSTM(10, activation='relu')(input1)
dense1 = Dense(10, activation='relu')(dense1)
dense1 = Dense(10, activation='relu')(dense1)
dense1 = Dense(10, activation='relu')(dense1)
dense1 = Dense(10, activation='relu')(dense1)

input2 = Input(shape=(3, 1 ))
dense2 = LSTM(10, activation='relu')(input2)
dense2 = Dense(10, activation='relu')(dense2)
dense2 = Dense(10, activation='relu')(dense2)
dense2 = Dense(10, activation='relu')(dense2)
dense2 = Dense(10, activation='relu')(dense2)

from keras.layers.merge import concatenate
merge1 = concatenate([dense1, dense2])

middle1 = Dense(30)(merge1)
middle1 = Dense(30)(middle1)
middle1 = Dense(30)(middle1)

output1 = Dense(30)(middle1)
output1 = Dense(30)(output1)
output1 = Dense(1)(output1) #3이 아니고 1

model = Model(inputs=[input1,input2], outputs=output1) #[output]>list형만 [], output'1'임

model.summary()


#3. 실행
model.compile (optimizer='adam', loss = 'mse')
model.fit([x1, x2], y, epochs=1000)
#ValueError: Error when checking model input: 
# the list of Numpy arrays that you are passing to your model is not the size the model expected.
#  Expected to see 2 array(s), but instead got the following list of 1 arrays
# model.fit(x1,x2,y,epochs=1000) x1,x2에 리스트 안걸어줌



#4. 평가

#############################################################
x1_predict = x1_predict.reshape(1,3,1)
x2_predict = x2_predict.reshape(1,3,1) #(3, ) //input_shape = (3,1)애 맞춰서 형태에 맞춰주기
#변수명만 변경해줬음

# x_input = array([5, 6, 7]) #평가를 해보고 싶어서 스칼라 3개, 벡터 1개 짜리 넣어줘
# x_input = x_input.reshape(1,3,1) #(3, ) //input_shape = (3,1)애 맞춰서 형태에 맞춰주기

#[[[5]
# [6]
# [7]]]

print(x1_predict)
print(x2_predict)

y_predict = model.predict([x1_predict,x2_predict])
print(y_predict)
