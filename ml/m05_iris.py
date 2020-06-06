# 다중분류 kneighbor// acc 확인하기


from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import LinearSVC, SVC
# 분류모델과 회귀모델을 각각 완성하시오
from sklearn.datasets import load_iris
# 1. data
iris = load_iris()
x = iris.data
y = iris.target

print(x.shape, y.shape) # (150, 4) (150,)

print(x)
print(y)

scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)

# scaler = StandardScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)


from sklearn.model_selection import train_test_split


x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8, shuffle = True)

# 2. model
# model = RandomForestClassifier() #0.96
# model = LinearSVC() # 0.93
# model = SVC() # 0.96
model = KNeighborsClassifier(n_neighbors = 1) # 0.96
# model = KNeighborsRegressor(n_neighbors = 1) # 0.94, acc 0.96
# model = RandomForestRegressor() # error
# 모델별로 acc 구하기

# 3. fit
model.fit(x_train, y_train)

# 4. predict

score = model.score(x_test, y_test)
y_predict = model.predict(x_test)
print('score', score)


from sklearn.metrics import accuracy_score, r2_score
acc = accuracy_score(y_test, y_predict)
R2 = r2_score(y_test, y_predict)
print('acc', acc)



# R2
# 1.회귀
# score와 R2 비교

# 3. 분류
# score와 accuracy_score 비교
