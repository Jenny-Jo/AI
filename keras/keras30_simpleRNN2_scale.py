##최적화 덜 됨
#shape이 가장 중요!!
#predict 50,60,70 맞추기

from numpy import array
from keras.models import Sequential
from keras.layers import SimpleRNN
from keras.layers.core import Dense

#1. Data
x = array([[1,2,3],[2,3,4], [3,4,5],[4,5,6],
          [5,6,7],[6,7,8],[7,8,9,],[8,9,10],
          [9,10,11],[10,11,12],
          [20,30,40],[30,40,50],[40,50,60]]) 
y = array([4,5,6,7,8,9,10,11,12,13,50,60,70]) #(13,3)
x_predict = array([50,60,70])


print("x.shape : ", x.shape)
print("y.shape : ", y.shape) ## .shape 중요!! ##

x = x.reshape(x.shape[0], x.shape[1], 1 ) 
print(x.shape)
print(x)


'''
              행         열         몇 개씩 자르는지
x의 shape = (batch_size, timesteps, feature)
batch size =행의 크기대로 자른다
feature은 열을 자른다

input_shape = (timestpes/열, feature)
              (input_length,input_dim)

'''
#2. 모델구성
model = Sequential()
model.add(SimpleRNN(500, input_length=3, input_dim=1)) 
model.add(Dense(500))
model.add(Dense(500))
model.add(Dense(500))
model.add(Dense(500))
model.add(Dense(500))
model.add(Dense(500))
model.add(Dense(500))
model.add(Dense(1))

model.summary()


#3. 실행
model.compile (optimizer='adam', loss = 'mse')


model.fit(x,y, epochs=1000)



#4. 평가
x_predict = x_predict.reshape(1,3,1) #(3, ) //input_shape = (3,1)애 맞춰서 형태에 맞춰주기
#변수명만 변경해줬음

# x_input = array([5, 6, 7]) #평가를 해보고 싶어서 스칼라 3개, 벡터 1개 짜리 넣어줘
# x_input = x_input.reshape(1,3,1) #(3, ) //input_shape = (3,1)애 맞춰서 형태에 맞춰주기

#[[[5]
# [6]
# [7]]]

print(x_predict)

y_predict = model.predict(x_predict)
print(y_predict)
#결과가 들쑥날쑥해서 잘하고 있는지 모르겠음
