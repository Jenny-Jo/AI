############################
#다중분류/one hot encoding###
############################

import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense
#중간에 넣어도 되나 맨 위에 넣는 것을 추천

#loss값, output layer 변경, y값 변경 되어 있어야
#1.  data

x = np.array(range(1, 11))
y = np.array([1,2,3,4,5,1,2,3,4,5])
#train, test /val

# 원핫인코딩
################################
from keras.utils import np_utils
y = np_utils.to_categorical(y)
################################

print(y) 
# [[0. 1. 0. 0. 0. 0.]
#  [0. 0. 1. 0. 0. 0.]
#  [0. 0. 0. 1. 0. 0.]
#  [0. 0. 0. 0. 1. 0.]
#  [0. 0. 0. 0. 0. 1.]
#  [0. 1. 0. 0. 0. 0.]
#  [0. 0. 1. 0. 0. 0.]
#  [0. 0. 0. 1. 0. 0.]
#  [0. 0. 0. 0. 1. 0.]
#  [0. 0. 0. 0. 0. 1.]]
# 모든 분류에 대한 가중치는 같다?!


# 1 ) 1열을 직접 잘라줘도 되고
print(y.shape) 
y = y[:, 1:]
print(y)
print(y.shape)

'''
# 2 ) 자료형으로 봐서 각 값에 1을 빼준다
import numpy as np
y = np.array([1,2,3,4,5,1,2,3,4,5])
y = y - 1

print(y) #[0 1 2 3 4 0 1 2 3 4]

from keras.utils import np_utils
y = np_utils.to_categorical (y) #to_categorical  이용하기 위해 무조건적으로 0으로 시작
print(y)
print(y.shape)
'''



#(10,0) >(10, 6) > 6 에서 5로, 0이 자동으로 인덱싱되는 문제. 0자리 없애서  바꾸기 !!!!숙제!!!!
print(x.shape) #(10,) #항상 shape 먼저 보기

#2. model

model = Sequential()

model.add(Dense(300, input_dim = 1, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(300)) #activation default 값 있어서, 없어도 돌아감
                               #########
model.add(Dense(5, activation='softmax')) ###
                               #########
#전지전능하신 activation은 모든 layer에 강림하신다!!
#sigmoid는 0 or 1로 수렴, 최종값에 곱해서 0이나 1이 나옴

model.summary()

#3.컴파일, 훈련
                                 ###############################              #######
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics= ['acc'])
                                 ################################             #######
model.fit(x, y, epochs=10, batch_size= 1)

#4.평가, 예측

loss,acc = model.evaluate(x, y, batch_size=1)
print("loss : ", loss)
print("acc : ", acc)

x_pred = np.array([1,2,3,4,5]) #output이 6개라서 하나 넣으면 6개 나와
y_predict = model.predict(x_pred)
print(y_predict)
y_predict = np.argmax(y_predict,axis=1) + 1 
# axis : 2차원에서 행(axis = 1)에서 최댓값을 빼었느냐, 열(axis = 0)에서 빼었느냐
print('y_pred: ',y_predict) 

#!!!!!!y_pred를 바꾸는 함수????!!
# y_pred:  [[0.08447615 0.18838052 0.18655092 0.18472885 0.16989201 0.18597163] 제일 큰 값  1
                # 0        1             2           3          4          5
#  [0.05172801 0.17869125 0.19075894 0.19404957 0.17365383 0.2111184 ]                     5
#  [0.03112592 0.16643532 0.19148125 0.20050484 0.17465068 0.23580205]]     
# 
# y_pred:  [[0.21092378 0.20677215 0.19483757 0.19502336 0.19244318] 1
#  [0.20905657 0.20650439 0.19460683 0.19570489 0.19412737] 1
#  [0.2025705  0.20501168 0.19430959 0.19903412 0.19907413]] 2

# y_pred:  [[0.21215552 0.20070623 0.2002933  0.19051124 0.19633374] 1
#  [0.19849712 0.19591473 0.20413128 0.19742124 0.20403571] 3
#  [0.18677607 0.19071889 0.20708707 0.20335016 0.21206777]] 5