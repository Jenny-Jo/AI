#1. 데이터_일부 전처리 해줌. 
import numpy as np

x1 = np.transpose([range(1,101), range(311,411)])
x2 = np.transpose([range(1,101),range(311,411)])

y1 = np.transpose([range(101,201), range(411,511)])
y2 = np.transpose([range(501,601),range(711,811)])
y3 = np.transpose([range(411,511),range(611,711)])



from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test = train_test_split ( 
    #x, y, random_state=66, shuffle=True,#default option, 원래대로 하고 싶으면 shuffle false
    x1, y1, shuffle= False,
    
    train_size=0.8
    )
# train(80,3), test (20,3)

from sklearn.model_selection import train_test_split
x2_train, x2_test, y2_train, y2_test, y3_train, y3_test = train_test_split ( 
    #x, y, random_state=66, shuffle=True,#default option, 원래대로 하고 싶으면 shuffle false
    x2, y2, shuffle= False,
    
    train_size=0.8
    )
  

#2. 모델구성/모델 두개
from keras.models import Sequential, Model #함수형 모델 땡겨옴
from keras.layers import Dense, Input



input1 = Input(shape=(2, )) #변수명은 소문자/행 무시, 열이 3개인 것만 명시/함수형은 input이 뭔지 명시해줌

dense1 = Dense(5, activation='relu',name='bitking1')(input1) #제일 앞의 것이 출력값/활성화 함수/맨 뒤에 앞단의 아웃풋이 input으로 지정
dense1 = Dense(100, activation='relu',name='bitking2')(dense1) #활성화함수 안써도 되는 이유; 디폴트가 있음, input, output 이름 똑같이 써도 됨
dense1 = Dense(100, activation='relu',name='bitking3')(dense1) #name parameter! 어떻게 연산이 되어있는지 summary에서 쉽게 볼 수 있도록 이름 부여
dense1 = Dense(300, activation='relu')(dense1) #활성화함수 안써도 되는 이유; 디폴트가 있음
dense1 = Dense(4, activation='relu')(dense1) 



input2 = Input(shape=(2, )) #변수명은 소문자/행 무시, 열이 3개인 것만 명시/함수형은 input이 뭔지 명시해줌

dense2 = Dense(5, activation='relu')(input2) #제일 앞의 것이 출력값/활성화 함수/맨 뒤에 앞단의 아웃풋이 input으로 지정
dense2 = Dense(100, activation='relu')(dense2) 
dense2 = Dense(100, activation='relu')(dense2) 
dense2 = Dense(4, activation='relu')(dense2) 
 #활성화함수 안써도 되는 이유; 디폴트가 있음

from keras.layers.merge import concatenate #사슬처럼 엮다/단순병합
merge1 = concatenate([dense1, dense2]) #두개 이상 값>list[]/이 자체가 레이어

middle1 = Dense(30)(merge1) #merge한 레이어를 인풋으로
middle1 = Dense(5)(middle1)
middle1 = Dense(7)(middle1)

######output model 구성######

output1 = Dense(30)(middle1)
output1_2 = Dense(50)(output1)
output1_3 = Dense(2)(output1_2) #아웃풋2!!!!

output2 = Dense(30)(middle1)
output2_2 = Dense(7)(output2)
output2_3 = Dense(2)(output2_2) #아웃풋2!!!!

output3 = Dense(30)(middle1)
output3_2 = Dense(7)(output3)
output3_3 = Dense(2)(output3_2) #아웃풋2!!!!


##model 명시##
model = Model(inputs=[input1,input2], 
             outputs=[output1_3,output2_3,output3_3]) #함수형 모델 범위 하단에 명시, 함수형모델 이름을 model 소문자로 씀/제일 처음, 제일 끝



model.summary() #모델 확인


#3. 훈련>weight
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
               #손실함수/   최적화함수/      판정방식 : 회귀방식 predict에 분류지표 accuracy가 들어가면 안됨.
model.fit([x1_train, x2_train], [y1_train, y2_train, y3_train], epochs=100, batch_size=2,
        verbose=1, validation_split= 0.5)
    #x_train : (60, 3), x_val : (30, 3), x_test : (20,3)
    #verbose가 0,1정도 되면 더 자세하게 보여주고 수치가 높아지면 간략화해서 보여줌f
          


#4. 평가,예측
loss = model.evaluate([x1_test, x2_test],[y1_test, y2_test, y3_test], batch_size=2) 
print("model.metrics_names : ", model.metrics_names) 

'''
반환값 5개
loss: 9.3246 - dense_12_loss: 5.0787 - dense_15_loss: 4.2459 - dense_12_mse: 5.0787 - dense_15_mse: 4.2459
전체/첫번째 output에 대한 loss/ 두번째 아웃풋에 대한 로스/ 메트릭스 mse 1/ 메트릭스 mse 2
'''
print("loss : ", loss)
# print("mse : ", mse)

# y_pred = model.predict(x_pred)
# print("y_predict : ", y_pred)


y1_predict,y2_predict,y3_predict = model.predict([x1_test,x2_test]) #(20,3)
# print([y1_predict,y2_predict,y3_predict]) #RMSE, R2 구하기 위해
# print("---------------")
# print(y1_predict)
# print("---------------")
# print(y2_predict)
# print("---------------")
# print(y3_predict)




#RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict)) #sqrt 는 루트
RMSE1 = RMSE(y1_test,y1_predict)
RMSE2 = RMSE(y2_test,y2_predict)
RMSE3 = RMSE(y3_test,y3_predict)


print("RMSE1: ", RMSE1)
print("RMSE2: ", RMSE2)
print("RMSE3: ", RMSE3)

print("RMSE : ", (RMSE1 + RMSE2 + RMSE3)/3)

#R2 구하기
from sklearn.metrics import r2_score 
r2_1 = r2_score(y1_test, y1_predict)
r2_2 = r2_score(y2_test, y2_predict)
r2_3 = r2_score(y3_test, y3_predict)

print("R2_1 : ", r2_1)
print("R2_2 : ", r2_2)
print("R2_3 : ", r2_3)

print("R2 :", (r2_1 + r2_2 + r2_3)/3)

#2개 들어가서 2개 나와?
