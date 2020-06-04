####################
# 이진분류 ##########
####################
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Input
#중간에 넣어도 되나 맨 위에 넣는 것을 추천

#1.  data

x = np.array(range(1, 11))
y = np.array([1,0,1,0,1,0,1,0,1,0])
#train, test /val

print(x.shape) #(10,) #항상 shape 먼저 보기
print(y.shape) #(10,) scalar 10개, vector가 1개 !=(10,1)

#2. model

model = Sequential()

model.add(Dense(100, input_dim = 1, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100)) #activation default 값 있어서, 없어도 돌아감
                               #########
model.add(Dense(1, activation='sigmoid'))
                               #########
#전지전능하신 activation은 모든 layer에 강림하신다!!
#sigmoid는 0 or 1로 수렴, 최종값에 곱해서 0이나 1이 나옴

model.summary()

#3.컴파일, 훈련
                                 ###############################
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics= ['acc'])
                                 ################################
model.fit(x, y, epochs=10, batch_size= 1)

#4.평가, 예측

loss,acc = model.evaluate(x, y, batch_size=1)
print("loss : ", loss)
print("acc : ", acc)

x_pred = np.array([1,2,3])
y_predict = model.predict(x_pred)
print('y_pred: ',y_predict)


# 1 ) y 값에 반올림 하기도 하고 
# y_predict = np.around(y_predict)
# print('y_pred: ',y_predict)

# y_predict.reshape(-1, 1)
#??????????

# y_predict = y_predict.reshape(y_predict.shape[0])
print(y_predict.shape)


# for i in range(len(y_predict)):
#     y_predict[i] = round(y_predict[i])

# print(y_predict)

# for i in range(len(y_pred)):
#     if y_pred[i]>0.5:
#         y_pred[i]=1
#     else:
#         y_pred[i]=0

# y_pre=[int(round(i)) for i in y_predict]
# # # 반올림
# print(y_pre)


# 2 ) 1차원 y를 2차원으로 reshape 해주어 ...?
from sklearn.preprocessing import OneHotEncoder
aaa = OneHotEncoder (  )
aaa.fit(y)
y = aaa.transform(y).toarray() #?????????????????????

print(y)
print(y.shape)