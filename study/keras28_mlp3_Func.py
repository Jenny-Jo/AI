#Keras16_mlp.py를 sequential에서 함수형으로 변경
#earlyStopping 적용

#1. 데이터_일부 전처리 해줌. 
import numpy as np
x = np.array(range(1,101)).T #1~100 (미만으로 보자!) 한두개 틀려도 돌아감. 
y = np.array([range(101,201), range(711,811), range(100)]).T #0~99

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split ( 
    #x, y, random_state=66, shuffle=True,#default option, 원래대로 하고 싶으면 shuffle false
    x, y, shuffle= False,
    
    train_size=0.8
    )


#2. 모델구성
from keras.models import Sequential, Model
from keras.layers import Dense, Input

input=Input(shape=(1 ,))

dense = Dense(10, activation='relu')(input)
dense = Dense(10, activation='relu')(dense)
dense = Dense(10, activation='relu')(dense)
dense = Dense(10, activation='relu')(dense)
dense = Dense(10, activation='relu')(dense)

output1 = Dense(30)(dense)
output1 = Dense(30)(output1)
output1 = Dense(30)(output1)
output1 = Dense(3)(output1)


model = Model (inputs = input, outputs=output1)

model.summary()

#3. 훈련>weight
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
               #손실함수/   최적화함수/      판정방식 : 회귀방식 predict에 분류지표 accuracy가 들어가면 안됨.

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=5, mode='auto)')
model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=3,
        validation_split=0.25, callbacks=[early_stopping])
          


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

