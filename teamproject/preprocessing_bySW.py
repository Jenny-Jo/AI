import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
from tqdm import tqdm
from glob import glob
from sklearn.model_selection import train_test_split
import csv

#Load-------------------------------------------------------------------------------------------------------------------
def load_images_from_folder(folder):
   images=[]
   for filename in tqdm(folder):
       img=cv2.imread(filename,cv2.IMREAD_COLOR) #컬러사진을 이용한다.
       img=img[15:200,30:300] #이미지를 얼굴부분만 나오게 잘라준다.
       images.append(img)
   images=np.array(images)
   return images


#특정데이터 추출
a=[]
def load(t):
   for root, dirs, files in os.walk(t):
       for fname in dirs:
           full_dirs = os.path.join(root, fname)0

           full = glob('{}/*.jpg'.format(full_dirs))
           if full != []:

               T=full
               try:
                   u = load_images_from_folder(T)
                   a.append(u)
               except:
                   print(T)

   u=np.asarray(a)
   return u

x=load('teamproject/test/data/E02')
#저장한다.
np.save('teamproject/test/data/E02FacesFeature.npy',arr=x)

-프로토타입 모델-
식별모델
import numpy as np
import pandas as pd
import os
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,BatchNormalization,GlobalAveragePooling2D,Activation,Reshape
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from ml_metrics import rmse

#Load--------------------------------------------------------------------------------------------
a=np.load('D:/s/koreaface/xa.npy')
# a=np.load('D:/s/koreaface/xb.npy')
# a=np.load('D:/s/koreaface/xc.npy')
# a=np.load('D:/s/koreaface/xd.npy')
## a=np.load('D:/s/koreaface/xe.npy')
# a=np.load('D:/s/koreaface/xf.npy')
# a=np.load('D:/s/koreaface/xg.npy')
# a=np.load('D:/s/koreaface/xh.npy')
# a=np.load('D:/s/koreaface/xi.npy')
# a=np.load('D:/s/koreaface/xj.npy')
# a=np.load('D:/s/koreaface/xk.npy')
# a=np.load('D:/s/koreaface/xl.npy')
# a=np.load('D:/s/koreaface/xn.npy')
# a=np.load('D:/s/koreaface/xm.npy')
# a=np.load('D:/s/koreaface/xo.npy')
# a=np.load('D:/s/koreaface/xp.npy')
# a=np.load('D:/s/koreaface/xq.npy')
# a=np.load('D:/s/koreaface/xr.npy')
#------------------------------------------------------------------------------------------------
#한 사람당 이미지 장수는 ? 10800개
#사이즈가 너무 커서 메모리 용량 부족-> 한개씩 가져와야된다.
#------------------------------------------------


print(a.shape)
a=a.reshape(10800,int(a.shape[0]*a.shape[1]/10800),100,143,3) # reshape를 사용해서 크기를 맞춰준다
#파일당 인원 체크
#1.=22 , =21 3.=18 4.=20 5.=?(스크래치),6.=20 ,7=24 ,8.=20,9.=23,10.=20 11.=23 1=21,13.=21,14=22,15=31,16.=25,17.=24,18.=23
#총 인원=334
print(a.shape)

k=np.arange(0,int(a.shape[1]),1) # K의 값은 순서에 따라 조정이 필요하다.
k=list(k)                        # 유일한 수동 조절부분)= Y값을 만들어주기 위해서 

u=int(a.shape[1]) #22
p=int(a.shape[0])
c=[]
for i in tqdm(k):        # X : Y (1:1) 대응으로 만들어 주는 부분
   for j in range(p):
       c.append(i)
y=np.asarray(c).reshape(-1,1)

#
x,x_pred,y,y_hat=train_test_split(a,y, train_size=0.9)
X=x.reshape(x.shape[0]*x.shape[1],x.shape[2],x.shape[3],x.shape[4])
X_pred=x_pred.reshape(x_pred.shape[0]*x_pred.shape[1],x_pred.shape[2],x_pred.shape[3],x_pred.shape[4])


output_size=int(x.shape[1])
# Y데이터는 사람의 이름이다
(이름대신 숫자값을 넣어준다. 숫자에서 이름을 바꿔주는 모형은 이미 미니 모델링에서 구현하였다.)
 X와 Y 1대1 대응 함수이다.

# 데이터사이즈 [None,100,143,3]

x_train,x_test,y_train,y_test=train_test_split(X,y,train_size=0.75)

# print(x_train.shape,x_test.shape)
# print(y_train.shape,y_test.shape)

#Model-Identification----------------------------------------------------------------------------
model=Sequential()
model.add(Conv2D(filters=64, kernel_size=(3,4),padding='valid',input_shape=(x_train.shape[1],x_train.shape[2],x_train.shape[3])))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=128, kernel_size=(2,3),padding='valid'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=512, kernel_size=(3,3),padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
# model.add=Dense(512,activation='relu')
model.add(Dense(output_size,activation='softmax')) # 여러명 중에서 이 사람이 맞는지 찾기위해(식별)
model.compile(loss='sparse_categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])

#Train_and_Pred----------------------------------------------------------------------------------
from keras.models import load_model

#first_ep---------------------------------------------------------------------------------------
model.fit(x_train,y_train,epochs=100,validation_split=(0.2))
model.evaluate(x_test,y_test)
model.save('./한안.h5')

#ep----------------------------------------------------------------------------------------------
# model = load_model('./한안.h5')
# model.fit(x_train,y_train,epochs=100,validation_split=(0.2))
# model.evaluate(x_test,y_test)
# model.save('./한안.h5')
#predict-----------------------------------------------------------------------------------------
# pred=model.predict(X_pred)

# print(pred,y_hat) #확인해본다.

검증모델

import numpy as np
import pandas as pd
import os
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,BatchNormalization,GlobalAveragePooling2D,Activation,Reshape
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from ml_metrics import rmse


#Load--------------------------------------------------------------------------------------------
a=np.load('D:/s/koreaface/xa.npy')
# a=np.load('D:/s/koreaface/xb.npy')
# a=np.load('D:/s/koreaface/xc.npy')
# a=np.load('D:/s/koreaface/xd.npy')
## a=np.load('D:/s/koreaface/xe.npy')
# a=np.load('D:/s/koreaface/xf.npy')
# a=np.load('D:/s/koreaface/xg.npy')
# a=np.load('D:/s/koreaface/xh.npy')
# a=np.load('D:/s/koreaface/xi.npy')
# a=np.load('D:/s/koreaface/xj.npy')
# a=np.load('D:/s/koreaface/xk.npy')
# a=np.load('D:/s/koreaface/xl.npy')
# a=np.load('D:/s/koreaface/xn.npy')
# a=np.load('D:/s/koreaface/xm.npy')
# a=np.load('D:/s/koreaface/xo.npy')
# a=np.load('D:/s/koreaface/xp.npy')
# a=np.load('D:/s/koreaface/xq.npy')
# a=np.load('D:/s/koreaface/xr.npy')
#------------------------------------------------------------------------------------------------
#한 사람당 이미지 장수는 ? 10800개
#사이즈가 너무 커서 메모리 용량 부족 한개씩 가져와야된다.

print(a.shape)
a=a.reshape(10800,int(a.shape[0]*a.shape[1]/10800),100,143,3)
#파일당 인원 체크
#1.=22 , =21 3.=18 4.=20 5.=?(스크래치),6.=20 ,7=24 ,8.=20,9.=23,10.=20 11.=23 1=21,13.=21,14=22,15=31,16.=25,17.=24,18.=23
#총 인원=334
print(a.shape)

k=np.arange(0,int(a.shape[1]),1) #K의 값은 순서에 따라 조정이 필요하다.(유일한 수동 조절부분)
k=list(k)
c=[]
o=0 #직접지정 : 총 334명
for i in k:
   for j in range(10800):
       if k.index(o)==i:
           k=1
           c.append(k)
       else:
           k=0
           c.append(k)
y=np.asarray(c).reshape(-1,1)

x,x_pred,y,y_hat=train_test_split(a,y, train_size=0.9)
X=x.reshape(x.shape[0]*x.shape[1],x.shape[2],x.shape[3],x.shape[4])
X_pred=x_pred.reshape(x_pred.shape[0]*x_pred.shape[1],x_pred.shape[2],x_pred.shape[3],x_pred.shape[4])


# #예측데이터를 10%떼어낸다. Train에는 전체데이터 90%로 사용한다.
# # X_pred=1080개
# # X=9720개
#
x_train,x_test,y_train,y_test=train_test_split(X,y,train_size=0.75)
#
# print(x_train.shape,x_test.shape)
# print(y_train.shape,y_test.shape)

#Model-Verification------------------------------------------------------------------------------
model=Sequential()
model.add(Conv2D(filters=64, kernel_size=(3,4),padding='valid',input_shape=(x_train.shape[1],x_train.shape[2],x_train.shape[3])))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=128, kernel_size=(2,3),padding='valid'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=512, kernel_size=(3,3),padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
# model.add=Dense(512,activation='relu')
model.add(Dense(1,activation='sigmoid'))  # 찾은 사람이 맞는지 (검증)
model.compile(loss='binary_crossentropy',optimizer='Adam',metrics=['accuracy'])
#Train_and_Pred---------------------------------------------------------------------------------
from keras.models import load_model
##first_ep--------------------------------------------------------------------------------------
model.fit(x_train,y_train,epochs=100,validation_split=(0.2))
model.evaluate(x_test,y_test)
model.save('./한안{}.h5'.format(o))

#ep---------------------------------------------------------------------------------------------# model = load_model('./한안{}.h5'.format(o))
# model.fit(x_train,y_train,epochs=100,validation_split=(0.2))
# model.evaluate(x_test,y_test)
# model.save('./한안{}.h5'.format(o))
#predict----------------------------------------------------------------------------------------
# pred=model.predict(X_pred)
# 다른 이미지를 구해 predict해본다.
# 모델이 돌아가는것까지는 확인했고, 최종 output은 모델 최적화를 한 후에 낼 예정입니다.



















