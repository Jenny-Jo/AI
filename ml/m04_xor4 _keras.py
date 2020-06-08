# input_dim = 1, output은 1, hidden 없이 바로 아웃풋

# 특정 모델빼곤 legacy machine learning 에선 잘 쓰지 않는다. 취업해서 나중에 훑어보기

from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.svm.classes import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor # 군집 분석
from keras.layers.core import Dropout
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# 1. Data
x_data = np.array([[0,0], [1, 0], [0, 1], [1, 1]])
y_data = np.array([0, 1, 1, 0])

print(x_data.shape) # (4, 2)
print(y_data.shape) # (4, )
# layer 마다 연산을 시키기 위해 리스트에서 넘파이로 해줌

# 2. Model
# model = LinearSVC() # 1.0
# model = SVC() # 1.0
# kn = KNeighborsClassifier(n_neighbors = 1) # 1.0
# model = KNeighborsRegressor(n_neighbors = 1) # 1.0
# model = RandomForestClassifier() # 1.0                    
model = RandomForestRegressor() # 1.0
# 이런식으로 임의로 정의해줌//
# logistic은 분류
# 모델 비교 해봐야 암

model = Sequential()

model.add(Dense(10, input_dim =2, activation='relu')) #앞에 1이 output
# model.add(Dense(1, activation='sigmoid')) 이 줄 추가하면 히든레이어 추가 되어 딥러닝
model.add(Dense(10, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))
model.summary()

# 이거 하나면 끝

# 3. 훈련 = compile + fit
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc']) #메트릭스는 훈련 X, 보여주는 값
model.fit(x_data, y_data, epochs=100, batch_size=1, verbose =1) # batch_size default 32
# 이진분류


# 4. Evaluate = score, Predict
x_test = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
# x_test와 x_data가 같아서 평가에  x_data 넣었다
y_predict = model.predict(x_test)
loss,acc = model.evaluate(x_data, y_data)

# acc score
print(x_test, "의 예측 결과: ", y_predict)
print("acc = ", acc)