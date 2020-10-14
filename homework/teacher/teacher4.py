import pandas as pd
from skimage import io
data = pd.read_csv('F:\Study\homework/teacher/a943287.csv')
print(data.head()) # 데이타 위에서 다섯번째 줄까지만 출력
print(data.shape)  # 데이터 shape 출력 (64084,10)
# print(data.columns())
print(data.describe()) # 데이터 평균, 표준편차,분산, 등 통계학적 수치를 보여주는 내적 함수
print(list(data)) # 데이터 열의 제목들을 리스트로 보여준다
print(len(list(data))) # 제목의 갯수

data_male = data[data['please_select_the_gender_of_the_person_in_the_picture']=='male'] # 데이터 중 남자의 데이터만 가져옴 
data_female = data[data['please_select_the_gender_of_the_person_in_the_picture']=='female'] # 데이터 중 여자의 데이터만 가져옴
final_data = pd.concat([data_male[:1000], data_female[:1000]], axis=0).reset_index(drop=True) # 여자, 남자의 데이터를  천 개씩 가져와 위 아래로 붙이고, 인덱스는 다시 오름차순으로 바꿈
print(final_data.shape)     # (2000, 10)
print("=====================1=====================")
# print(final_data.loc['image_url'])
print(final_data.iloc[0]) # final_data의 첫번째 행(인덱스)의 데이터를 보여줌
print("=====================2=====================")
print(final_data.iloc[1])
print("======================3====================")
print(final_data.iloc[2])
print("======================4====================")
# print(final_data[0])
print(type(final_data)) # final_data의 데이터 타입이 판다스임을 알려줌. pandas.core.frame.DataFrame
print("======================5====================")
print(final_data.loc[0]['image_url']) # 0번째 행의 'image_url'의 열의 위치의 정보를 알려줌  
print(final_data.loc[0][0]) # 0번째 행의 0번째 열의 정보를 알려줌 (1023132475)
print(final_data.loc[0][7])# 0번째 행의 7번째 의 열의 위치의 정보를 알려줌  = 'image_url' # 'https://d1qb2nb5cznatu.cloudfront.net/users/40-large'
print("=====================6=====================")
print(final_data) # final data를 간략히 보여줌
print("=====================7=====================")
print(final_data.iloc[0][7])
print(final_data.iloc[0]['image_url'])
print(final_data.iloc[0].loc['image_url'])
print(final_data.iloc[0].iloc[7])
# 전부다 url의 위치를 찾아 url을 보여줌

# 판다스에서 iloc, loc 는 행으로 자료의 위치를 찾고 iloc는 위치정수를 기반으로 인덱싱하고, loc는 레이블을 기반으로 인덱싱한다

print("=====================8=====================")
print(final_data.loc[0]['please_select_the_gender_of_the_person_in_the_picture'])
# 첫번째 행의 성별을 찾아줌 : male

from skimage import io # 이미지를 읽는 방법 중 하나 / Image reading via the ImageIO Library
from matplotlib import pyplot as plt
img = io.imread(final_data.loc[0]['image_url']) # 첫째 행의 url로 이미지 읽어와서 넘파이로 변환
print(img.shape)        # (300, 300, 3)
io.imshow(img)   # 읽어온 이미지를 보여준다

print('data_male', data_male)
print('여기까지 왔다.')

data_2000 = io.imread('./teacher\down')
import os
path_list = ['./teacher/train', './teacher/test']
for path in path_list:
    os.makedirs(path)


train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set', target_size =(64,64), batch_size=32, class_mode='binary')
test_set = test_datagen.flow_from_directory('dataset/test_set', target_size=(64,64), batch_size=32, class_mode='binary')


from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential()
model.add(Conv2D(32,(3,3),input_shape = (50,50,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(32,(3,3),input_shape = (50,50,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(units =128, activation='relu'))
model.add(Dense(units=1, activation='sigmoid' ))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit_generator(training_set, steps_per_epoch=300, epochs=25, validation_data=test_set, validation_steps=2000)


####
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras import optimizers, initializers, regularizers, metrics
from tensorflow.keras.preprocessing.image import ImageDataGenerator

np.random.seed(3)
tf.random.set_seed(3)

train_datagen = ImageDataGenerator(rescale=1./255,          # 0~1변환 / 정규화한다.
                                   horizontal_flip=True,    # vertical_flip : 주어진 이미지를 수평 또는 수직으로 뒤집는다.
                                   width_shift_range=0.1,   # 정해진 범위 안에서 그림을 수평 또는 수직으로 랜덤하게 평행 이동
                                   height_shift_range=0.1,  #
                                   # rotation_range         #  정해진 각도만큼 이미지를 회전
                                   fill_mode='nearest'      # 이미지를 축소 또는 회전하거나 이동할때 새익는 빈 공간을 어떻게 할지.
                                                            # nearest  가장 비슷한 색으로 채운다.
                                   )

train_generator = train_datagen.flow_from_directory(
    './data/train',
    target_size=(150, 150),
    batch_size=160,           # 원래 5
    class_mode='binary'
)

# print(train_generator)
print("===================================")
# print(train_generator[0])
# print(train_generator[0].shape)   # error
print(train_generator[0][0].shape)  # (160, 150, 150, 3)
print(train_generator[0][1].shape)  # (160,)
print("===================================")
print(len(train_generator))         # 32

# 테스트셋은 이미지 부풀리기 과정을 진행하지 않음
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    './data/test',
    target_size=(150, 150),
    batch_size=200,               # 원래 5
    class_mode='binary'
)

print("test_x.shape : ", test_generator[0][0].shape)  # (120, 150, 150, 3)
print("test_y.shape : ", test_generator[0][1].shape)  # (120,)

np.save('./data/train_x.npy', arr=train_generator[0][0])
np.save('./data/train_y.npy', arr=train_generator[0][1])
np.save('./data/test_x.npy', arr=test_generator[0][0])
np.save('./data/test_y.npy', arr=test_generator[0][1])




# 앞서 배운 CNN 모델을 만들어 적용하기

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(150, 150, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Activation('sigmoid'))

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=optimizers.Adam(learning_rate=0.0002),
              metrics=['accuracy'])

history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=20,
    validation_data=test_generator,
    validation_steps=4
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
y_vloss = history.history['val_loss']
y_loss = history.history['loss']

print("우갸갹")

print(data_male)
