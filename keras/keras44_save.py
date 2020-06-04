# 튜닝 해야함!!
# keras40_lstm_split1.py
# earlystopping
# LSTM 모델을 완성하시오

import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM
from array import array



#2.모델구성
model = Sequential()

model.add(LSTM(4, input_shape = (4, 1), activation='relu'))

model.add(Dense(4))
model.add(Dense(4))
model.add(Dense(1))

model.summary()
 
############################
# model.save("save_44.h5") #  저장폴더 지정 안해주면 기본 폴더에 저장됨
# model.save(".//model//save_keras44.h5")
model.save("./model/save_keras44.h5")
# model.save(".\model\save_keras44.h5")
# 세 개 다 됨

############################
print("저장 잘됨")

