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

samsung = np.load('./test_samsung.py/samsung.npy', allow_pickle=True)
hite = np.load('./test_samsung.py/hite.npy', allow_pickle=True)

# print(samsung.shape)
# samsung = samsung.reshape(509, )
# print(samsung.shape)
# samsung = (split_x(samsung, size))


# 데이터 전처리 4 // Standard_scaler, PCA
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(hite)
hite_scaled = scaler.transform(hite)

# 삼성전자도 표준화 해주기 > 나중에 예측값 predict 할 때 값 안나오지 않음?


from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(hite_scaled)


# hite_pca = pca.transform(hite) 
# print(hite_pca.shape) #(509,1)
 
hite = pca.transform(hite) # 변수명 자주바꾸면 헷갈리므로 자주 바꾸지말기
print(hite.shape)


# 데이터 전처리 5 // split 함수
#samsung
size = 6
def split_a(seq, size) : 
    aaa=[]
    for i in range(len(seq)-size+1) :
        subset = seq[i:(i+size)]
        aaa.append([item for item in subset])
    print(type(aaa))
    return np.array(aaa)

# samsung1 = split_a(samsung, size) # 변수명 자주바꾸면 헷갈리므로 자주 바꾸지말기
samsung = split_a(samsung, size)
print('samsung: ', samsung)
print(samsung.shape) # (504,6,1)

samsung = samsung.reshape(504,6)

# 데이터 전처리 6 // x, y 나누기

x1 = samsung[:, :-1]
y = samsung[:, -1]
print('x1: ', x1)    
# print('y: ', y)      
print(x1.shape, y.shape) # (504, 5) (504,)


# 
x2 = hite
print(x2.shape) #(509, 2)
x2 = x2[ : -5, :]
print(x2.shape) #(504, 2)



print('------------5----------------------')

# 데이터 전처리 7 // train, test 나누기

from sklearn.model_selection import train_test_split
'''
x1_train, x1_test= train_test_split(x1, shuffle = False,train_size=0.8)

print(x1_train.shape) # (403, 5)
print(x1_test.shape)  # (101, 5)

x2_train, x2_test= train_test_split(x2, shuffle = False,train_size=0.8)

print(x2_train.shape)  # (403, 2)
print(x2_test.shape)   # (101, 2)

y_train, y_test= train_test_split(y, shuffle = False,train_size=0.8)

print(y_train.shape)  
print(y_test.shape)  
'''

x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(x1, x2, y, shuffle=False, train_size=0.8)

# 2. model 앙상블 구성 lstm!!
from keras.models import Sequential, Model
from keras.layers import Dense, Input

input1 = Input(shape =(5, ))
dense1 = Dense(300, activation='relu')(input1)
dense1 = Dropout(0.2)(dense1)


input2 = Input(shape =(2, ))
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


#modelpath='./model/{epoch:02d} - {val_loss:.4f}.hdf5'
#checkpoint = ModelCheckpoint(filepath= modelpath, monitor='val_loss', save_best_only=True, mode = 'auto')


#tensorboard = TensorBoard(log_dir ='graph', histogram_freq=0, write_graph=True, write_images=True)


hist = model.fit([x1_train,x2_train], y_train, epochs =1, batch_size=100, verbose=1,
                validation_split=0.4, 
                callbacks=[earlystopping]) #checkpoint,tensorboard])

# model.save('./test_samsung.py/test0602_HyunJung.save.h5')



# 4. 평가, 예측 / 시각화
loss_mse = model.evaluate([x1_test, x2_test], y_test, batch_size=60)

# loss = hist.history['loss']
# val_loss = hist.history['val_loss']
# mse = hist.history['mse']
# val_mse = hist.history['val_mse']

# print('mse: ', mse)
# print('val_mse: ', val_mse)

print('loss, mse : ',loss_mse )

# 예측 
# -1 을 넣는 것이 맞는지 잘모르겠어요 내일 선생님께 물어볼게요
# 2020-06-02,"51,000"
# 2020-06-01,"50,800"
# 2020-05-29,"50,000"
# 2020-05-28,"51,100"
# 2020-05-27,"48,950"

# x1_predict = x1_test[-1] # 6/1 까지 samsung 데이터
# x2_predict = x2_test[-1] # 6/1 까지 hite 데이터 

# print(x1_predict.shape) # (5, )
# print(x2_predict.shape) # (2, )

# x1_predict = x1_predict.reshape(1, 5)  # input 1과 동일한 shape (5, ) 행무시
# x2_predict = x2_predict.reshape(1, 2)  # input 2와 동일한 shape(2, )


y_predict = model.predict([x1_test, x2_test])
print('y_predict : ', y_predict)


for i in range(5) :
  print('마지막날 시가: ', y_test[i], '/예측가: ', y_predict[i])

# RMSE 구하기
'''
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y1_pred))
# R2 구하기
from sklearn.metrics import r2_score

r2 = r2_score(y_test, y1_pred)
print("R2 : ", r2)
'''