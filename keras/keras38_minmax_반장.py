from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM




#데이터 

x = array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], 
           [5, 6, 7], [6, 7, 8], [7, 8, 9], [8, 9, 10],
           [9, 10, 11], [10, 11, 12], 
           [2000, 3000, 4000], [3000, 4000, 5000], [4000, 5000, 6000], [100,200,300]])  

y = array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 5000, 6000, 7000, 400])  #(14,) 



x_predict = array([55,65,75])


# print("x.shape : ", x.shape)    # (14,3)
# print("y1.shape : ", y.shape)  # (14, )    


from sklearn.preprocessing import MinMaxScaler, StandardScaler


scaler1 = MinMaxScaler()
scaler1.fit(x)
x = scaler1.transform(x)

x_predict = x_predict.reshape(-1,3)  
x_predict = scaler1.transform(x_predict)

'''
  뭐지 에러 뜨는데...?
ValueError: Expected 2D array, got 1D array instead:
array=[55. 65. 75.].
Reshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample.
나중에 해결하자   와꾸조심 
x_predict = x_predict.reshape(-1,3) 로 바꾸니 해결
[[173.52254]]   [[126.8457]]
혹은 train_data = np.reshape(train_data, shape=(-1, num_features)) 를 사용해서 
각 column에서 스케일링이 진행되게 할 수도 있다. 
물론 모델 들어가기 전에 다시 reshape는 필수 
'''
print(x_predict)

# scaler2 = StandardScaler()
# scaler2.fit(x)
# x = scaler2.transform(x)

'''
print(x) 
scaler1 = MinMaxScaler()
[[0.00000000e+00 0.00000000e+00 0.00000000e+00]
 [2.50062516e-04 2.00080032e-04 1.66750042e-04]
 [5.00125031e-04 4.00160064e-04 3.33500083e-04]
 [7.50187547e-04 6.00240096e-04 5.00250125e-04]
 [1.00025006e-03 8.00320128e-04 6.67000167e-04]
 [1.25031258e-03 1.00040016e-03 8.33750208e-04]
 [1.50037509e-03 1.20048019e-03 1.00050025e-03]
 [1.75043761e-03 1.40056022e-03 1.16725029e-03]
 [2.00050013e-03 1.60064026e-03 1.33400033e-03]
 [2.25056264e-03 1.80072029e-03 1.50075038e-03]
 [4.99874969e-01 5.99839936e-01 6.66499917e-01]
 [7.49937484e-01 7.99919968e-01 8.33249958e-01]
 [1.00000000e+00 1.00000000e+00 1.00000000e+00] 
 [2.47561890e-02 3.96158463e-02 4.95247624e-02]]
________________________________________________________
scaler2 = StandardScaler()
[[-0.5091461  -0.52172253 -0.52813466]
 [-0.50836632 -0.52112564 -0.52765244]
 [-0.50758653 -0.52052876 -0.52717022]
 [-0.50680674 -0.51993187 -0.526688  ]
 [-0.50602695 -0.51933498 -0.52620578]
 [-0.50524716 -0.51873809 -0.52572356]
 [-0.50446737 -0.51814121 -0.52524134]
 [-0.50368759 -0.51754432 -0.52475912]
 [-0.5029078  -0.51694743 -0.5242769 ]
 [-0.50212801 -0.51635054 -0.52379468]
 [ 1.04965084  1.26774696  1.39930025]
 [ 1.82943921  1.86463471  1.88152064]
 [ 2.60922758  2.46152247  2.36374103]
 [-0.43194706 -0.40353876 -0.38491521]]
0을 기준으로 양 옆으로 쫙 퍼져있음 
'''


x = x.reshape(x.shape[0], x.shape[1], 1)   #x.shape :  (13, 3, 1)
x_predict = x_predict.reshape(1,3,1)


# 2. 모델 구성

model = Sequential()
model.add(LSTM(15, activation='relu', input_shape = (3,1)))  

# model.add(LSTM(10, input_length = 3 ,input_shape = 1))
model.add(Dense(12))
model.add(Dense(12))
model.add(Dense(12))
model.add(Dense(1))


# 3. 실행 

from keras.callbacks import EarlyStopping
ealry_stopping= EarlyStopping(monitor='loss', patience= 50,  mode = 'auto') 


model.compile(optimizer='adam', loss = 'mse')
model.fit(x,y, epochs= 10000, callbacks= [ealry_stopping], verbose= 2)


y_predict = model.predict(x_predict)
print(y_predict)     



'''
[[96.55454]]
데이터가 [1,2,3,..13, 14, 400,500,600] 이렇게 편중된 데이터가 있다 보니까 예측값도 치우쳐서 나타난다 
데이터 전처리 필요 -> wiki 10일차에 작성
'''

'''
그런데 standard scaling을 했더니 에러가 나거나 y_predict 값이 [[125829.54]] 이렇게 정신 나간 값이 나온다! MinMax scaling도 마찬가지
왜 그럴까?
-> x_predict 값이 스케일링이 안되어 있었기 때문
-> 마찬가지로 같은 전처리를 해줘야 한다. 
-> 단, 따로 하게 되면 서로 다른 스케일을 따라서 소용이 없다.
그렇다면?
scaler.fit(x)를 실행하게 되면 scaler는 계산치? 가중치를 저장하게 된다
그래서 
x_predict = scaler.transform(x_predict)
를 추가해준다. 
'''
