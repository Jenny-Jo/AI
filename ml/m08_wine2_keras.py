
# ------------------------
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import LinearSVC, SVC
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

# 1. data
# 1) 불러오기
wine = pd.read_csv("./ml/winequality-white.csv", index_col =None, header = 0, sep=';') 
print(wine)
print(wine.shape) #(4898, 12)
wine = wine.values # 판다스에서 넘파이로 바꿔야함

# 2) 스케일링/이상치 제거
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(wine)
wine_scaled = scaler.transform(wine)

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(wine_scaled)

# 3) x, y 나누기
print(type(wine))
x = wine[ : , :-1]
y = wine[ : ,  -1]

print(x.shape) # (4898, 11)
print(y.shape) # (4898, )

# 4) 다중분류 y OneHotEncoding
from keras.utils import np_utils
y = np_utils.to_categorical(y)

print(y.shape) # (4898, 10)

# 4) train, test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, train_size=0.8)

# 2. model

model = Sequential()

model.add(Dense(11, input_shape = (11, ), activation = 'relu'))
model.add(Dense(10, activation ='relu'))
model.add(Dense(10, activation ='relu'))
model.add(Dense(10, activation ='relu'))
model.add(Dense(10, activation = 'softmax'))

# 3. compile fit
model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics =['acc'] )
model.fit(x_train, y_train, batch_size = 30, epochs=30, validation_split= 0.3)

# 4. evaluation 없이 predict
loss, acc = model.evaluate(x_test, y_test, batch_size =100)
y_predict = model.predict(x_test)

