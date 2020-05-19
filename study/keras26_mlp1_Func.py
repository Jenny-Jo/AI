#Keras14_mlp.py를 sequential에서 함수형으로 변경
#earlyStopping 적용

#1. 데이터_일부 전처리 해줌. 
import numpy as np
x = np.transpose([range(1,101), range(311,411), range(100)]) #1~100 (미만으로 보자!) 한두개 틀려도 돌아감. 
y = np.array([range(101,201), range(711,811), range(100)]).T #0~99

print(x.shape) #(100,3)

# print(x.shape) 3행 100열 

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split ( 
    #x, y, random_state=66, shuffle=True,#default option, 원래대로 하고 싶으면 shuffle false
    x, y, shuffle= False, ###True가 좋다!!
    train_size=0.8
    )


#2. 모델구성
from keras.models import Sequential, Model
from keras.layers import Dense, Input

input1 = Input(shape=(3, ))

dense1= Dense (5, activation='relu')(input1)
dense1= Dense (40, activation='relu')(dense1)
dense1= Dense (40, activation='relu')(dense1)

output= Dense(40)(dense1)
output= Dense(40)(output)
output= Dense(3)(output)


model = Model(inputs= input1, outputs= output)

model.summary()

#3. 훈련>weight
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
               #손실함수/   최적화함수/      판정방식 : 회귀방식 predict에 분류지표 accuracy가 들어가면 안됨.
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=5,mode ='auto')
model.fit(x_train, y_train, epochs=100, batch_size=1,
        validation_split=0.25, callbacks=[early_stopping]),
    #x_train : (60, 3), x_val : (20, 3), x_test : (20,3)
          


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

