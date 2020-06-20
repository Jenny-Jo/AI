import urllib.request
from konlpy.tag import Okt
from gensim.models.word2vec import Word2Vec
import pandas as pd
import matplotlib.pyplot as plt
from keras.utils import np_utils
from gensim.models import Word2Vec
from keras import preprocessing
from keras.engine.sequential import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.core import Activation, Dense, Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling1D
from keras.layers.recurrent import LSTM
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence
import gensim
import re


# Y 욕
y_data = pd.read_csv('F:\\Study\\miniproject\y_data\\욕취합.csv',index_col=None, header =None, sep='\t')



# while True : 
#     line = y_data.readline()
#     line = line.replace('\n','')
#     y_data.append(line)

#     if not line : break
# y_data.close()


# X 기사 댓글들
a = pd.read_csv('F:\\Study\\miniproject\\x_data\\1.csv', index_col=0, header =0, sep='\t')
b = pd.read_csv('F:\\Study\\miniproject\\x_data\\2.csv',index_col=0, header =0, sep='\t')

replys = pd.concat([a, b], axis =0)

label = [0]*len(replys.values)
my_reply_dic = {'reply':[], 'label' : label}


j = 0

for reply in replys:
    reply_data = reply.text
    reply_data = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…\"\“》]','',reply_data)
    my_reply_dic['reply'].append(reply_data)


    for i in range(len(y_data)):
        slangflag = False
        if reply_data.find(y_data[i]) != -1:
            slangflag = True
            print(i, '악플?''테스트 : ',reply_data.find(y_data[i]), '비교단어 : ', '인덱스 : ', i, reply_data)
            break
    if slangflag == True  : 
        label[j] = 1
    elif slangflag == False :
        label[j] = 0
    j = j + 1

my_reply_dic['label'] = label
my_reply_df = pd.DataFrame(my_reply_dic)

def dftoCsv(my_reply_df, num) : 
    my_reply_df.to_csv(('F:\Study\miniproject'+str(num) + '.csv'), sep=',', na_rep='NaN', encoding='utf-8')


x = x_data.loc[:,"댓글 내용"]#.values)
print(y_data.shape) # (5, 21)
print(x.shape) # ( 5913, )

print(type(y_data))
print(y_data.head())

y_data = y_data.values
y_data = y_data.reshape(-1,1)



# ValueError: invalid literal for int() with base 10: ' 좃찐'

print(x.isnull().values.any())              # Null값 존재 유무 : False
print(len(x))                               # 5913

x= x.str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")   # 정규 표현식을 통한 한글 외 문자 제거

# #토큰의 사이즈를 정의를 해준다
# def tokenize(doc):

#     return ['/'.join(t) for t in pos_tagger.pos(doc, norm=True, stem=True)]


# ## training Word2Vec model using skip-gram   
# tokens = [tokenize(row[1]) for row in x]
# model = gensim.models.Word2Vec(size=300,sg = 1, alpha=0.025,min_alpha=0.025, seed=1234)
# model.build_vocab(tokens)

# #불용어 정의
# stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']


# 형태소 분석기 

from eunjeon import Mecab
tagger = Mecab()
out = pd.DataFrame(data = None,index=None,columns=["댓글"])
# print(x)
# idx = 0
for i in x:
    tmp = pd.Series([tagger.morphs(i)],index=["댓글"])
    out = out.append(tmp,ignore_index= True)
    # idx += 1

# print(x)
# okt = Okt()
# tokenized_data = []
# for sentence in x:
#     temp_X = okt.morphs(sentence, stem=True) # 토큰화
#     temp_X = [word for word in temp_X if not word in stopwords] # 불용어 제거
#     tokenized_data.append(temp_X)

# word to vec
from gensim.models.word2vec import Word2Vec
import re
model = Word2Vec(sentences = x[0], size = 100, window = 5, min_count = 5, workers = 4, sg = 0)
model1 = Word2Vec(sentences = y_data[0], size = 100, window = 5, min_count = 5, workers = 4, sg = 0)

print(model.wv.vectors.shape)           # (911, 100)

# print(model.wv.most_similar('김여정')) # KeyError: "word '김여정' not in vocabulary"

y = np_utils.to_categorical(y_data)

x_train, x_test = train_test_split(x,test_size = 0.8)
y_train, y_test = train_test_split(y, test_size = 0.8)

x_train = sequence.pad_sequences(x_train, maxlen=100)
x_test = sequence.pad_sequences(x_test, maxlen=100)
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

print(x_train.shape)
print( x_test.shape)
print(y_data.shape)

# (2456, 3)
# (3459, 3)
# (50, 21)
# (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=5000)

maxlen=20
x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)

model = Sequential()
model.add(Embedding(5000,100))
model.add(Dropout(0.5))
model.add(Conv1D(64,5, padding= 'valid', activation='relu', strides=1))
model.add(MaxPooling1D(pool_size=4))
model.add(LSTM(55))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(x_train, y_train, batch_size=100, epochs=5, validation_data=(x_test,y_test))

#테스트 정확도 출력
print("\n 정확도 : %.4f" % (model.evaluate(x_test, y_test)[1]))

#테스트셋의 오차
y_vloss = history.history['val_loss']

#학습셋의 오차
y_loss = history.history['loss']

x_len = np.arange(len(y_loss))
plt.plot(x_len, y_vloss, marker='.', c="red", label="Testset_loss")
plt.plot(x_len, y_loss, marker='.', c="blue", label="Trainset_loss")

#그래프에 그리드를 주고 레이블을 표시
plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
