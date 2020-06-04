#verbose = 학습의 진행 상황을 얼마나 보여줄 것인가

#1. 데이터_일부 전처리 해줌. 
import numpy as np
x = np.array([range(1,101), range(311,411), range(100)]) #1~100 (미만으로 보자!) 한두개 틀려도 돌아감. 
y = np.array(range(101,201)) #0~99,

x = np.transpose(x)
y = np.transpose(y)

print(x.shape) #(100, 3)
print(y.shape)
# print(x.shape) 3행 100열 (3,100)



from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split ( 
    #x, y, random_state=66, shuffle=True,#default option, 원래대로 하고 싶으면 shuffle false
    x, y, shuffle= False,
    
    train_size=0.8
    )
# train (30,3), test (20,3)

    

#2. 모델구성
from keras.models import Sequential, Model #함수형 모델 땡겨옴
from keras.layers import Dense, Input #함수형 모델은 input layer 명시해야 함

input1 = Input(shape=(3, )) #변수명은 소문자/행 무시, 열이 3개인 것만 명시/함수형은 input이 뭔지 명시해줌/함수형에서는 shape사용

dense1 = Dense(5, activation='relu')(input1) #제일 앞의 것이 출력값/활성화 함수/맨 뒤에 앞단의 아웃풋이 input으로 지정/activation default = linear
dense1 = Dense(100, activation='relu')(dense1) #활성화함수 안써도 되는 이유; 디폴트가 있음(linear), input, output 이름 똑같이 써도 됨
dense1 = Dense(100, activation='relu')(dense1) 
dense1 = Dense(100, activation='relu')(dense1) 
dense1 = Dense(100, activation='relu')(dense1) 
dense1 = Dense(100, activation='relu')(dense1) 
dense1 = Dense(100, activation='relu')(dense1) 
dense1 = Dense(100, activation='relu')(dense1) 
dense1 = Dense(100, activation='relu')(dense1) 

output1 = Dense(1)(dense1)

model = Model(inputs=input1, outputs=output1) #함수형모델 범위 설정, 함수형모델 이름을 model 소문자로 씀

model.summary() #모델 확인


#3. 훈련>weight
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
               #손실함수/   최적화함수/      판정방식 : 회귀방식 predict에 분류지표 accuracy가 들어가면 안됨.
model.fit(x_train, y_train, epochs=100, batch_size=1,
        validation_split=0.25, verbose=2
        )
    #x_train : (60, 3), x_val : (30, 3), x_test : (20,3)
    #verbose가 0,1정도 되면 더 자세하게 보여주고 수치가 높아지면 간략화해서 보여줌
          


#4. 평가,예측
loss, mse = model.evaluate(x_test, y_test, batch_size=1) 
    #같은 값으로 예측하면 안됨. 훈련데이타, 평가 데이타는 달라야 함. 여기서 predict 값 생성.

print("loss : ", loss)
print("mse : ", mse)

# y_pred = model.predict(x_pred)
# print("y_predict : ", y_pred)????왜 x_pred이 아닌 x_test로 y_predict를 구했나??

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

