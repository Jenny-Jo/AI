from keras.applications import VGG16, VGG19, Xception, ResNet101, ResNet101V2, ResNet152
from keras.applications import ResNet152V2, ResNet50, ResNet50V2, InceptionV3, InceptionResNetV2
from keras.applications import MobileNet, MobileNetV2, DenseNet121, DenseNet169, DenseNet201
from keras.applications import NASNetLarge, NASNetMobile

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, BatchNormalization, Activation, Input
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.datasets import cifar10
from keras import optimizers
import numpy as np



# (None, 244, 244, 3)
# vgg16.summary()


(x_train,y_train),(x_test, y_test) = cifar10.load_data()
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

input_tensor = Input(shape=(32, 32, 3))

model = Sequential()
vgg16 = VGG16(include_top = False, weights ='imagenet', input_tensor = input_tensor)
model.add(vgg16)
model.add(Flatten())
model.add(Dense(256))
model.add(Dense(10, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer = optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics = ['accuracy'])

model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=32, epochs=3)

model.save_weights('./model/param_vgg.hdf5')

scores = model.evaluate(x_test, y_test, verbose=1)
print('test loss', scores[0])
print('test accuracy', scores[1])

pred = np.argmax(model.predict(x_test[0:10]), axis=1)
print(pred)
'''
test loss 0.6731343188285828
test accuracy 0.7709000110626221'''