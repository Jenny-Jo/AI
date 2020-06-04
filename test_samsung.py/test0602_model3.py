# lstm 2개 구현
import numpy as np
from keras.models import Model, load_model
from keras.layers import Dense, LSTM, Dropout, Input
from keras.layers.merge import concatenate, Concatenate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

def split_x(seq, size) : 
    aaa=[]
    for i in range(len(seq)-size+1) :
        subset = seq[i:(i+size)]
        aaa.append([item for item in subset])
    print(type(aaa))
    return np.array(aaa)

size = 6

# 1. Data
# npy 불러오기

samsung = np.load('./test_samsung.py/samsung.npy', allow_pickle='Ture')
hite = np.load('./test_samsung.py/hite.npy', allow_pickle='True')

print(samsung.shape) # (509, 1)
print(hite.shape)    # (509, 5)

samsung = samsung.reshape(samsung.shape[0], ) # (509, ) 
samsung = (split_x(samsung, size))
print(samsung.shape) #  (504, 6)

print("------------------------------")
x_sam = samsung[:, 0:5]
y_sam = samsung[:, 5]

print(x_sam.shape)   # (504, 5)
print(y_sam.shape)   # (504, )

x_hit = hite[5:510, : ] #앙상블에서 shape 고쳐야 함

print(x_hit.shape)   # (504, 5)

x_sam = x_sam.reshape(504,5,1)
x_hit = x_hit.reshape(504,5,1)




print("-----------------------------------")

# 2. 모델구성(LSTM) dropout 도 넣기
 
input1 = Input(shape=(5,1))
x1 = LSTM(10, return_sequences=False)(input1)
x1 = Dense(10)(x1)

input2 = Input(shape=(5,1))
x2 = LSTM(10, return_sequences=False)(input2)
x2 = Dense(10)(x2)
x2 = Dropout(0.2)(x2)

merge = concatenate([x1, x2])
merge = Dense(10)(merge)

output = Dense(10)(merge)
output = Dense(1)(output)

model = Model(inputs= [input1,input2], outputs=output)

model.summary()

# 3. compile, fit
model.compile(loss='mse', optimizer='adam', metrics=['mse'] )
model.fit([x_sam, x_hit], y_sam, epochs=1)



