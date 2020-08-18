import dlib, cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
# 전처리





# 모델구성
model = Sequential()
model.add(Conv2D(50, (2,2), input_shape = (28,28, 1)))
model.add(Conv2D(50,(5,5)))
model.add(Dropout(0.3))
model.add(Conv2D(50,(2,2)))
model.add(MaxPooling2D(pool_size=4))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))
model.summary()

# 실행
from keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='loss', patience=10, mode='auto')

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=20, batch_size=256, verbose=1, callbacks=[early_stopping])

# 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=256)

print('loss: ', loss)
print('acc: ', acc)
