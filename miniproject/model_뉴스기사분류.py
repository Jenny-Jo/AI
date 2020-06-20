import os, json, glob, sys, numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import Dense, Embedding, LSTM, Flatten, Dropout, Input, Conv1D, MaxPooling1D, Bidirectional, GlobalMaxPool1D
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint
import keras.backend.tensorflow_backend as K
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

root_dir = './dataset/신문기사자료'
categories = ['100', '101', '102', '103', '104', '105']
nb_classes = len(categories)

X = []
y = []


for c_idx, cat in enumerate(categories):
    dir_detail = root_dir + "/" + cat
    files = glob.glob(dir_detail+"/*.txt.*")
    
    for i, fname in enumerate(files):
        with open(fname, "r", encoding='utf-8') as f:
            for idx, content in enumerate(f):
                X.append(content)
                y.append(c_idx)
        if i % 900 == 0:
            print(cat, " : ", fname)
            
            
print(len(X))
print(X[0])

y = np_utils.to_categorical(y, nb_classes)
print(y)


max_word = 5000
max_len = 300

tok = Tokenizer(num_words = max_word)
tok.fit_on_texts(X)

sequences = tok.texts_to_sequences(X)
print(len(sequences[0]))
print(sequences[0])

sequences_matrix = sequence.pad_sequences(sequences, maxlen=max_len)
print(sequences_matrix)
print(sequences_matrix[0])
print(len(sequences_matrix[0]))


print(len(tok.word_index))

X_train, X_test, y_train, y_test = train_test_split(sequences_matrix, y, test_size=0.2)

print(X_train.shape)
print(y_train.shape)

with K.tf_ops.device('/device:GPU:0'):
    model = Sequential()
    
    model.add(Embedding(max_word, 64, input_length=max_len))
    model.add(LSTM(60, return_sequences=True))
    model.add(GlobalMaxPool1D())
    model.add(Dropout(0.2))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model_dir = './model'
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    model_path = model_dir + "/predict_korea_news_LSTM.model"
    checkpoint = ModelCheckpoint(filepath=model_path, monitor="val_loss", verbose=1, save_best_only=True)

    early_stopping = EarlyStopping(monitor='val_loss', patience=7)

model.summary()

hist = model.fit(X_train, y_train, batch_size=128, epochs=20, validation_split=0.2, callbacks=[checkpoint, early_stopping])

print("정확도 : %.4f" % (model.evaluate(X_test, y_test)[1]))

y_vloss = hist.history['val_loss']
y_loss = hist.history['loss']

x_len = np.arange(len(y_loss))

plt.plot(x_len, y_vloss, marker='.', c='red', label='val_set_loss')
plt.plot(x_len, y_loss, marker='.', c='blue', label='train_set_oss')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('loss')
plt.grid()
plt.show()


y_vloss = hist.history['val_acc']
y_loss = hist.history['acc']

x_len = np.arange(len(y_loss))

plt.plot(x_len, y_vloss, marker='.', c='red', label='val_set_loss')
plt.plot(x_len, y_loss, marker='.', c='blue', label='train_set_oss')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('loss')
plt.grid()
plt.show()