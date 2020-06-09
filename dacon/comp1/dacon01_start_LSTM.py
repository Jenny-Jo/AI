import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
from keras.losses import MeanAbsoluteError

# y_pred.to_csv(경로) >> submit 할 파일 만든다
# 회귀
# 인풋 71개 > y output 4개
# 평가는 mae / loss, metrics(mse or mae)에 포함

# 1. Data

# 1) 불러오기
train = pd.read_csv('./data/dacon/comp1/train.csv', header = 0, index_col = 0)
test = pd.read_csv('./data/dacon/comp1/test.csv', header = 0, index_col = 0)
submission = pd.read_csv('./data/dacon/comp1/sample_submission.csv', header = 0, index_col = 0) # sep=',' 이게 디폴트

# 2) shape
print('train.shape : ', train.shape)
print('test.shape : ', test.shape)
print('submission.shape : ', submission.shape)

#  train.    shape :  (10000, 75) : 71 열은 x_train, x_test// 4 열은 y_train,y_test
#   test.    shape :  (10000, 71) : x_predict 
# submission.shape :  (10000, 4)  : y_predict 4

# 3) 결측치 보완
# print(train.isnull().sum())
# 3-1) interpolate
train = train.interpolate() #  보간법// 선형보간 -값들 전체의 선을 그리고 빈 자리를 선에 맞게 그려줌
test = test.interpolate() #  보간법// 선형보간 -값들 전체의 선을 그리고 빈 자리를 선에 맞게 그려줌

# 3-2) 첫행도 채워주기
train = train.fillna(method='bfill') 
test = test.fillna(method='bfill') 

# 4) x, y 나눠주기
x1 = train.values[:, :71].reshape(10000,71,1)
y1 = train.values[:, 71:]
x_predict = test.values.reshape(10000,71,1)
y_predict = submission.values


# # 5) Standard scaler, PCA
# scaler = StandardScaler()
# scaler.fit(x1)
# x1_scaled = scaler.transform(x1)
# x_predict = scaler.transform(x_predict)



# from sklearn.decomposition import PCA
# pca = PCA(n_components=2)
# pca.fit(x1_scaled)
# x1 = pca.transform(x1_scaled)
# x_predict = pca.transform(x_predict)

x_train, x_test, y_train, y_test = train_test_split(x1, y1, train_size = 0.8)


# 2. model

model = Sequential()
model.add(LSTM(10, input_shape=(71, 1), return_sequences=True))
model.add(LSTM(10, input_shape=(71, 1)))
model.add(Dense(1000))
model.add(Dropout(0.2))
model.add(Dense(4))

model.summary()

# 3. compile, fit
model.compile(optimizer = 'adam',loss = 'mae', metrics = ['mae'])
model.fit(x_train, y_train, epochs= 30, batch_size =10, validation_split = 0.3)

# 4. evaluation, predict
loss, mae = model.evaluate(x_test, y_test, batch_size=1) 
y_predict = model.predict(x_predict)
print('loss,mae:  ', loss,mae)
print(y_predict.shape)
 

y_pred = pd.DataFrame({
    'id' : np.array(range(10000, 20000)),
    'hhb': y_predict[:,0 ],
    'hbo2': y_predict[:,1],
    'ca': y_predict[:,2 ],
    'na': y_predict[:,3],
})
print(y_pred)

y_pred.to_csv('./dacon/y_pred.csv', index = False)
model.save('./dacon/dacon1_model.save.h5')



