from sklearn.svm import LinearSVC  # 회귀 except)logistic linear만 분류     # 선형분류에 특화
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor # 군집 분석
                             #          분류                  회귀
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from keras.layers.core import Dropout

#1. data
x_data = np.array([[0, 0],[1, 0],[0, 1],[1,1]])             # (4, 2)
y_data = np.array([0, 1, 1, 0])                              # (4, )


#2. model 
# model = LinearSVC()                                   
model = Sequential()
model.add(Dense(1, input_shape =(2, ), activation = 'sigmoid'))
model.add(Dense(1000, activation = 'sigmoid'))
model.add(Dropout(0.3))
model.add(Dense(1000, activation = 'sigmoid'))
model.add(Dropout(0.3))
model.add(Dense(1, activation = 'sigmoid'))

#3. fit
# model.fit(x_data, y_data)
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['acc'])  
model.fit(x_data, y_data, epochs =300, batch_size =1)
#4. evaluate, predict
# x_test = [[0, 0], [1, 0], [0, 1],[1, 1]]
# y_predict = model.predict(x_test)

                     
# acc = accuracy_score([0, 1, 1, 0], y_predict)        # evaluate = score()
#                      #  y_test

# print(x_test, '의 예측 결과: ', y_predict)
# print('acc = ', acc)
# [[0, 0], [1, 0], [0, 1], [1, 1]] 의 예측 결과:  [0 1 1 0]    
# add =  1.0

x_test = np.array([[0, 0], [1, 0], [0, 1],[1, 1]])
y_test = np.array([0, 1, 1, 0])
     
loss, acc = model.evaluate(x_test, y_test, batch_size = 1)                    

y_predict = model.predict(x_test)
y_predict = np.where( y_predict > 0.5, 1, 0)

print(x_test, '의 예측 결과: ', y_predict)

print('acc = ', acc)