from numpy import array
from keras.models import Model             
from keras.layers import Dense, LSTM, Input

#1. Data
x = array([[1,2,3],[2,3,4], [3,4,5],[4,5,6],
          [5,6,7],[6,7,8],[7,8,9],[8,9,10],
          [9,10,11],[11,12,13],
          [2000,3000,4000],[3000,4000,5000],[4000,5000,6000],
          [100,200,300]])# 마지막 줄 데이터를 10씩 곱해서 나열해줌  # (14,3)


y = array([4,5,6,7,8,9,10,11,12,13,5000,6000,7000,400]) # (14, )


x_predict = array([55,65,75])              # (3, )

#############################################################
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# scaler = MinMaxScaler()
scaler = StandardScaler()
scaler.fit(x)           # fit x 넣어서 전처리 실행
x = scaler.transform(x) # Transform 변환
x_predict = scaler.transform(x_predict)
print(x)
print(x_predict)
'''
MinMaxSclaer
[[0.00000000e+00 0.00000000e+00 0.00000000e+00]
 [2.50062516e-04 2.00080032e-04 1.66750042e-04]
 [5.00125031e-04 4.00160064e-04 3.33500083e-04]
 [7.50187547e-04 6.00240096e-04 5.00250125e-04]
 [1.00025006e-03 8.00320128e-04 6.67000167e-04]
 [1.25031258e-03 1.00040016e-03 8.33750208e-04]
 [1.50037509e-03 1.20048019e-03 1.00050025e-03]
 [1.75043761e-03 1.40056022e-03 1.16725029e-03]
 [2.00050013e-03 1.60064026e-03 1.33400033e-03]
 [2.50062516e-03 2.00080032e-03 1.66750042e-03]
 [4.99874969e-01 5.99839936e-01 6.66499917e-01]
 [7.49937484e-01 7.99919968e-01 8.33249958e-01]
 [1.00000000e+00 1.00000000e+00 1.00000000e+00]
 [2.47561890e-02 3.96158463e-02 4.95247624e-02]]

0과 1사이로 압축됨

StandardScaler

[[-0.50921604 -0.52177665 -0.52817863]
 [-0.50843623 -0.52117975 -0.5276964 ]
 [-0.50765642 -0.52058285 -0.52721417]
 [-0.5068766  -0.51998595 -0.52673194]
 [-0.50609679 -0.51938904 -0.52624972]
 [-0.50531698 -0.51879214 -0.52576749]
 [-0.50453717 -0.51819524 -0.52528526]
 [-0.50375736 -0.51759834 -0.52480303]
 [-0.50297755 -0.51700144 -0.5243208 ]
 [-0.50141793 -0.51580764 -0.52335634]
 [ 1.04962448  1.26773222  1.39929104]
 [ 1.82943464  1.8646331   1.88152013]
 [ 2.6092448   2.46153399  2.36374921]
 [-0.43201483 -0.40359027 -0.38495659]]

 0을 기준으로 해서 좌우로 쫙 펴짐 -0.38495659 이게 평균, 2.36374921 이게 거의 맨끝값
'''


# print("x1.shape : ", x1.shape)              # (13, 3)
# print("y.shape : ", y.shape)                # (13, )# .shape 중요!! ##

# x1 = x1.reshape(x1.shape[0], x1.shape[1], 1 ) #(13, 3, 1)

# print(x1.shape)
# print(x1)


# '''
#               행         열         몇 개씩 자르는지
# x의 shape = (batch_size, timesteps, feature)
# batch size =행의 크기대로 자른다
# feature은 열을 자른다

# input_shape = (timestpes/열, feature)
#               (input_length,input_dim)

# '''
#2. 모델구성
from keras.models import Sequential, Model
from keras.layers import Dense
model = Sequential()

# ############################################################################

# model.add(LSTM (10, input_length=3, input_dim=1, return_sequences=True)) 
# (13,3,1) 행무시 > input_dim = feature 1 > (none, 3, 1)
# Dense는 2차원을 받아들이고, LSTM은 3차원 받아들임 / return_squences는 2차원을 3차원으로 변환

# model.add(LSTM(10, return_sequences= False))

# ############################################################################

model.add(Dense(10))
model.add(Dense(1))


model.summary()

# '''
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# lstm_1 (LSTM)                (None, 3, 10)             480

# # 1개 input dim 1에서 output node 10개 (이게 output shape에서 feture됨. (None, 3, 10)), 480개로 증폭됨

# _________________________________________________________________
# lstm_2 (LSTM)                (None, 10)                840

# # 840 = 4* (10+1 bias + 10)
# # output node의 갯수는 그 다음 시작의 feature 이다.

# _________________________________________________________________
# dense_1 (Dense)              (None, 10)                110
# _________________________________________________________________
# dense_2 (Dense)              (None, 1)                 11
# =================================================================
# Total params: 1,441
# Trainable params: 1,441
# Non-trainable params: 0
# _________________________________________________________________


# '''



# ##################################################################################
# '''
# #3. 실행
model.compile (optimizer='adam', loss = 'mse')

from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss', patience=5, mode='auto')
model.fit(x, y, epochs=300, callbacks = [es])


# #4. 평가
x_predict = x_predict.reshape(1,3,1) #(3, ) //input_shape = (3,1)애 맞춰서 형태에 맞춰주기

# #변수명만 변경해줬음

# x_input = array([5, 6, 7]) #평가를 해보고 싶어서 스칼라 3개, 벡터 1개 짜리 넣어줘
# x_input = x_input.reshape(1,3,1) #(3, ) //input_shape = (3,1)애 맞춰서 형태에 맞춰주기

#[[[5]
# [6]
# [7]]]

print(x_predict)
y_predict = model.predict(x_predict)
print(y_predict)

# #결과가 들쑥날쑥해서 잘하고 있는지 모르겠음
# '''