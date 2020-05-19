#1. 데이터_일부 전처리 해줌. 
import numpy as np
x1 = np.array([range(1,101), range(311,411), range(100)]).T #1~100 (미만으로 보자!) 한두개 틀려도 돌아감. 
y1 = np.array([range(101,201),range(711,811), range(100)]).T #0~99,

x2 = np.array([range(101,201), range(411,511), range(100,200)]).T
y2 = np.array([range(501,601),range(711,811), range(100)]).T 


#weight=1, bias는 각자 다름
#(3,100)에서 (100,3)으로 변경
#인풋 아웃풋 3개씩


from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test = train_test_split ( 
    #x, y, random_state=66, shuffle=True,#default option, 원래대로 하고 싶으면 shuffle=false
    x1, y1, shuffle= False,
    train_size=0.8
    )
# train(80,3), test (20,3)

x2_train, x2_test, y2_train, y2_test = train_test_split ( 
      x2, y2, shuffle= False,
      train_size=0.8
    )
    
# ----------------------------------------------------------------------------------------------------------------------------
#2. 모델구성/모델 두개
from keras.models import Sequential, Model #함수형 모델 땡겨옴
from keras.layers import Dense, Input


#### input model ######
input1 = Input(shape=(3, )) #변수명은 소문자/행 무시, 열이 3개인 것만 명시/함수형은 input이 뭔지 명시해줌

dense1 = Dense(5, activation='relu',name='bitking1')(input1) #제일 앞의 것이 출력값/활성화 함수/맨 뒤에 앞단의 아웃풋이 input으로 지정
dense1 = Dense(100, activation='relu',name='bitking2')(dense1) #활성화함수 안써도 되는 이유; 디폴트가 있음, input, output 이름 똑같이 써도 됨
dense1 = Dense(100, activation='relu',name='bitking3')(dense1) #name parameter! 어떻게 연산이 되어있는지 summary에서 쉽게 볼 수 있도록 이름 부여
dense1 = Dense(300, activation='relu')(dense1) 
dense1 = Dense(4, activation='relu')(dense1) 



input2 = Input(shape=(3, )) 

dense2 = Dense(5, activation='relu')(input2)
dense2 = Dense(100, activation='relu')(dense2) 
dense2 = Dense(100, activation='relu')(dense2) 
dense2 = Dense(4, activation='relu')(dense2) 

##### merge model ######
from keras.layers.merge import concatenate #사슬처럼 엮다/단순병합
merge1 = concatenate([dense1, dense2]) #두개 이상 값>list[]/이 자체가 레이어/parameter연산 이루어지지 않음

middle1 = Dense(30)(merge1) #merge한 레이어를 인풋으로
middle1 = Dense(5)(middle1)
middle1 = Dense(7)(middle1)

###### output model ######

output1 = Dense(30)(middle1)
output1_2 = Dense(50)(output1)
output1_3 = Dense(3)(output1_2) #아웃풋3

output1 = Dense(30)(middle1)
output2_2 = Dense(7)(output1)
output2_3 = Dense(3)(output2_2) #아웃풋3

###함수형 모델 범위 명시###
model = Model(inputs=[input1,input2], 
             outputs=[output1_3,output2_3]) #함수형 모델 범위 하단에 명시, 함수형모델 이름을 model 소문자로 씀/제일 처음, 제일 끝



model.summary() #모델 확인/두 모델의 layer가 번갈아 나옴

# ----------------------------------------------------------------------------------------------------------------------------
#3. 훈련>weight
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
               #손실함수/   최적화함수/      판정방식 : 회귀방식 predict에 분류지표 accuracy가 들어가면 안됨.
model.fit([x1_train, x2_train], [y1_train, y2_train], epochs=100, batch_size=2,
        verbose=1
        )
    #x_train : (60, 3), x_val : (30, 3), x_test : (20,3)
    #verbose가 0,1정도 되면 더 자세하게 보여주고 수치가 높아지면 점점 간략화해서 보여줌
          

# ----------------------------------------------------------------------------------------------------------------------------
#4. 평가,예측
loss = model.evaluate([x1_test, x2_test],[y1_test, y2_test], batch_size=2) 
print("model.metrics_names : ", model.metrics_names)
#왜 loss만 명기하고 mse는 빼나? too many values to unpack(expected2)
'''
반환값 5개
loss: 9.3246 - dense_12_loss: 5.0787 - dense_15_loss: 4.2459 - dense_12_mse: 5.0787 - dense_15_mse: 4.2459
전체/첫번째 output에 대한 loss/ 두번째 아웃풋에 대한 로스/ 메트릭스 mse 1/ 메트릭스 mse 2
'''
print("loss : ", loss)
# print("mse : ", mse)

# y_pred = model.predict(x_pred)
# print("y_predict : ", y_pred)


y1_predict,y2_predict = model.predict([x1_test,x2_test]) #(20,3)
print([y1_predict,y2_predict]) #RMSE, R2 구하기 위해
print("---------------")
print(y1_predict)
print("---------------")
print(y2_predict)



#RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict)) #sqrt 는 루트
RMSE1 = RMSE(y1_test,y1_predict)
RMSE2 = RMSE(y2_test,y2_predict)

print("RMSE1 : ", RMSE1)
print("RMSE2: ", RMSE2)
print("RMSE : ", (RMSE1 + RMSE2)/2)


#R2 구하기
from sklearn.metrics import r2_score 
r2_1 = r2_score(y1_test, y1_predict)
r2_2 = r2_score(y2_test, y2_predict)
print("R2_1 : ", r2_1)
print("R2_2 : ", r2_2)
print("R2 :", (r2_1 + r2_2)/2)

