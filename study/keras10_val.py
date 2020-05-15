#1. 데이터
import numpy as np
x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,2,3,4,5,6,7,8,9,10])
x_test = np.array([11,12,13,14,15])
y_test = np.array([11,12,13,14,15])
# x_pred = np.array([16,17,18])
x_val = np.array ([101, 102, 103, 104, 105])
y_val = np.array ([101, 102, 103, 104, 105])
#유추할 수 없는 W값 생성


#predict
#One Data [1~15]에서 train 1~10, test 11~15. train과 test 분리를 알아서 해야 함.


#2. 모델구성
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()


model.add(Dense(5, input_dim = 1))
# model.add(Dense(1000000))
# model.add(Dense(1000000))
# model.add(Dense(1000000))
# model.add(Dense(1000000))
# model.add(Dense(1000000))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))

model.add(Dense(1))

model.summary()

#3. 훈련>weight
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
               #손실함수/   최적화함수/      판정방식 : 회귀방식 predict에 분류지표 accuracy가 들어가면 안됨.
model.fit(x_train, y_train, epochs=200, batch_size=1,
          validation_data=(x_val, y_val))
          


#4. 평가,예측
loss, mse = model.evaluate(x_test, y_test, batch_size=1) 
    #같은 값으로 예측하면 안됨. 훈련데이타, 평가 데이타는 달라야 함. 여기서 predict 값 생성.

print("loss : ", loss)
print("mse : ", mse)

# y_pred = model.predict(x_pred)
# print("y_predict : ", y_pred)

y_predict = model.predict(x_test)
print(y_predict)

#RMSE 구하기
from sklearn.metrics import mean_squared_error
#함수정의 def ,함수이름은 사용자 정의
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict)) #sqrt 는 루트
print("RMSE : ", RMSE (y_test, y_predict))

#R2 구하기
from sklearn.metrics import r2_score 
r2= r2_score(y_test, y_predict) #r2_predict
print("R2 : ", r2)


