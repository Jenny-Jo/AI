# 1. 6/3 내일아침 삼성전자 시가 맞추기/ 시계열 예측 RNN
# 2. CSV 데이터는 건들지 말 것/날짜 거꾸로 된 것 바꾸기
# 3. 앙상블 모델 사용
# 4. 체크포인트/얼리스타핑/텐서보드/ 전부 다 응용하기
# 5. 6/2 오늘 저녁 6시까지 메일
# 하이트진로와 삼성전자 CSV 데이타
# 앙상블은 행은 같게, 열은 다르게

# 1. Data -CSV load하기/standard scaler/ pca /x, y 나누고/ train_test_split
import numpy as np
import pandas as pd
from keras.layers.core import Dropout

# 데이터 전처리 1//data 불러와서 dropna, hite는 bfill 까지
# 1
print('hite')
hite = pd.read_csv("./test_samsung.py/hite_stock.csv", index_col=0, header=0,sep=',')
print(hite)
print(hite.shape)       #(720,5)
print(hite.values)


hite = hite.dropna(axis=0, how='all')
hite = hite.fillna(method='bfill') 
print(hite.shape)        #(509,5)
print('hite: ', hite)


# 2
print('samsung')
samsung = pd.read_csv("./test_samsung.py/samsung_stock.csv", index_col=0, header=0, sep=',')
print(samsung)
print(samsung.shape)   #(700, 1)


samsung = samsung.dropna(axis = 0 )
print(samsung.shape)       #(509, 1)



# 데이터 전처리 2 // 문자형에서 실수화
for i in range(len(hite.index)):
    for j in range(len(hite.iloc[i])):
        hite.iloc[i,j] = int(hite.iloc[i,j].replace(',', ''))

for i in range(len(samsung.index)):
    for j in range(len(samsung.iloc[i])):
        samsung.iloc[i,j] = int(samsung.iloc[i,j].replace(',', ''))

# 데이터 전처리 3 // 일자 오름차순으로 정리
hite = hite.sort_values(['일자'], ascending=[True])
samsung = samsung.sort_values(['일자'], ascending=[True])
print(hite)
print(samsung)

# 데이터 값 저장
hite = hite.values
samsung = samsung.values

print(hite)
print(samsung)
print(type(hite)) # <class 'numpy.ndarray'> #(508, 5)
print(type(samsung)) # <class 'numpy.ndarray'> #(509, 1)

np.save('./test_samsung.py/hite.npy', arr = hite)
np.save('./test_samsung.py/samsung.npy', arr = samsung)

hite = hite

print("---------4--------------")

# 데이터 전처리 4 // Standard_scaler, PCA
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(hite)
hite_scaled = scaler.transform(hite)

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(hite_scaled)
hite_pca = pca.transform(hite)
print(hite_pca.shape) #(509,1)
 




# 데이터 전처리 5 // split 함수
samsung
size = 6
def split_a(seq, size) : 
    aaa=[]
    for i in range(len(seq)-size+1) :
        subset = seq[i:(i+size)]
        aaa.append([item for item in subset])
    print(type(aaa))
    return np.array(aaa)

samsung1 = split_a(samsung, size)
print('samsung1: ', samsung1)
print(samsung1.shape) # (504,6,1)

samsung1 = samsung1.reshape(504,6)

# 데이터 전처리 6 // x, y 나누기

x1 = samsung1[:, :-1]
y = samsung1[:, -1]
print('x1: ', x1)    
# print('y: ', y)      
print(x1.shape, y.shape) # (504, 5) (504,)


# 
x2 = hite
print(x2.shape) #(509, 5)
x2 = x2[ : -5, :]
print(x2.shape) #(504,5 )



print('------------5----------------------')

# 데이터 전처리 7 // train, test 나누기
from sklearn.model_selection import train_test_split
x1_train, x1_test= train_test_split(x1, train_size=0.8)

print(x1_train.shape) 
print(x1_test.shape)  

x2_train, x2_test= train_test_split(x2, train_size=0.8)

print(x2_train.shape)
print(x2_test.shape)  

y_train, y_test= train_test_split(y, train_size=0.8)

print(y_train.shape)  
print(y_test.shape)  


# 2. model 앙상블 구성 lstm!!
from keras.models import Sequential, Model
from keras.layers import Dense, Input, LSTM

input1 = Input(shape =( 5, ))
dense1 = Dense(300, activation='relu')(input1)
dense1 = Dropout(0.2)(dense1)


input2 = Input(shape =( 5, ))
dense2 = Dense(300, activation='relu')(input2)
dense2 = Dropout(0.2)(dense2)


from keras.layers.merge import concatenate
merge1 = concatenate([dense1, dense2])
middle1 = Dense(100)(merge1)
middle1 = Dropout(0.2)(middle1)

output1 = Dense(100)(middle1)
output1 = Dense(1)(output1)

model = Model(inputs=[input1, input2], outputs=[output1])
model.summary()


# 3. 컴파일,훈련/callbacks-ES, Tensorboard, ModelCheckpoint
model.compile(loss='mse', optimizer='adam', metrics=['mse'])

from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

earlystopping= EarlyStopping( monitor = 'loss', patience = 20 )

modelpath='./model/{epoch:02d} - {val_loss:.4f}.hdf5'
checkpoint = ModelCheckpoint(filepath= modelpath, monitor='val_loss', save_best_only=True, mode = 'auto')

tensorboard = TensorBoard(log_dir ='graph', histogram_freq=0, write_graph=True, write_images=True)


hist = model.fit([x1_train,x2_train], y_train, epochs =30, batch_size=100, verbose=1,
                validation_split=0.4, 
                callbacks=[earlystopping, checkpoint,tensorboard])

model.save('./test_samsung.py/test0602_HyunJung.save.h5')



# 4. 평가, 예측 / 시각화
loss_mse = model.evaluate([x1_test, x2_test], y_test, batch_size=60)

loss = hist.history['loss']
val_loss = hist.history['val_loss']
mse = hist.history['mse']
val_mse = hist.history['val_mse']

print('mse: ', mse)
print('val_mse: ', val_mse)

print('loss, mse : ',loss_mse )




y1_pred = model.predict([x1_test,x2_test])

for i in range(5) :
    print('마지막날 시가: ', y_test[i], '/예측가: ', y1_pred[i])

# plt

import matplotlib.pyplot as plt
plt.figure(figsize=(10,7))


# 1
plt.subplot(2,1,1)
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')

# 2
plt.subplot(2,1,2)
plt.plot(hist.history['mse'])
plt.plot(hist.history['val_mse'])
plt.grid()
plt.title('mse')
plt.ylabel('mse')
plt.xlabel('epoch')
plt.legend(['mse', 'val_mse'])

plt.show()


# RMSE 구하기

from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y1_pred))

# R2 구하기
from sklearn.metrics import r2_score

r2 = r2_score(y_test, y1_pred)
print("R2 : ", r2)

