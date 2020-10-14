
# ImageDataGenerator로 남녀 구분 모델 만들기

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



# 1
train_generator = train_datagen.flow_from_directory(
    './teacher/down/train',
    target_size=(150, 150),
    batch_size=160,           # 원래 5
    class_mode='binary'
)

# print(train_generator)
print("=====================1================")
# print(train_generator[0])
# print(train_generator[0].shape)   # error
print(train_generator[0][0].shape)  # (160, 150, 150, 3) 사진들
print(train_generator[0][1].shape)  # (160,)??????????????라벨
print("======================2===============")
print(len(train_generator))         # 11 # 32??????????

# 2. 테스트셋은 이미지 부풀리기 과정을 진행하지 않음
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    './teacher/down/test',
    target_size=(150, 150),
    batch_size=200,               # 원래 5
    class_mode='binary'
)

print("test_x.shape : ", test_generator[0][0].shape)  # (200, 150, 150, 3)
print("test_y.shape : ", test_generator[0][1].shape)  # (200,)

np.save('./data/train_x.npy', arr=train_generator[0][0])
np.save('./data/train_y.npy', arr=train_generator[0][1])
np.save('./data/test_x.npy', arr=test_generator[0][0])
np.save('./data/test_y.npy', arr=test_generator[0][1])

print


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

import matplotlib.pyplot as plt
'''
def show_graph(history_dict):
    accuracy = history_dict['accuracy']
    val_accuracy = history_dict['val_accuracy']
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']
    
    epochs = range(1, len(loss) + 1)
    
    plt.figure(figsize=(16, 1))
    plt.subplot(121)
    plt.subplots_adjust(top=2)
    plt.plot(epochs, accuracy, 'ro', label='Training accuracy')
    plt.plot(epochs, val_accuracy, 'r', label='Validation accuracy' )
    plt.title('Training and validation accuracy and loss')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy and Loss')

    plt.legend(loc = 'upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=5)
    
    plt.subplot(122)
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper center', bbox_to_anchor = (0.5, -0.1),
               fancybox=True, shadow=True, ncol=5)
    plt.show()
show_graph(history.history)
'''