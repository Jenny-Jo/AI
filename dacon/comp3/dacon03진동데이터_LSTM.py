import numpy as np
import pandas as pd
from keras.layers import LSTM, Dense, Dropout, Conv1D,Flatten, MaxPooling1D
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 1. data

# 1) 불러 오기
test_features = pd.read_csv("./data/dacon/comp2/test_features.csv",  header = 0, index_col = 0)
train_features = pd.read_csv("./data/dacon/comp2/train_features.csv", header = 0, index_col = 0)
train_target = pd.read_csv("./data/dacon/comp2/train_target.csv", header = 0, index_col = 0)

# 2) shape
print(train_features)   # (1050000, 5)
print(train_target)     # (2800, 4)
print(test_features)    # (262500, 5)




# 3) x, y 나눠주기
x1 = train_features.values.reshape(2800,375,5)
y1 = train_target               # (2800, 4)
x_predict = test_features.values.reshape(700, 375, 5)

# 4) Standard scaler, PCA

x_train, x_test, y_train, y_test = train_test_split(x1, y1, train_size =0.8)

# 2. model

model = Sequential()
model.add(LSTM(200, input_shape = (375, 5), activation='relu', return_sequences=True))
model.add(LSTM(200, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(4))

model.summary()

# 3. compile, fit
model.compile(optimizer='adam', loss='mse', metrics=['mse'])
model.fit(x_train,y_train, epochs=30, batch_size=10, validation_split=0.2)

# 4. evaluation, predict
loss, mse = model.evaluate(x_test,y_test, batch_size=10)
y_predict = model.predict(x_predict)
print('loss,mse : ', loss, mse)
print(y_predict.shape)

y_pred = pd.DataFrame({
    'id' : np.array(range(2800, 3500)),
    'X'  : y_predict[:,0],
    'Y'  : y_predict[:,1],
    'M'  : y_predict[:,2],
    'V'  : y_predict[:,3]
})
print(y_pred)

y_pred.to_csv('./dacon/comp3/comp3_y_pred.csv')









