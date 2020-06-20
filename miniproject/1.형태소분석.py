from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, Dropout
from keras.preprocessing import sequence
from keras.utils import np_utils

import numpy
import tensorflow as tf
import matplotlib.pyplot as p0lt
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling1D
from keras.layers.core import Activation
import pandas as pd
from sklearn.model_selection import train_test_split
from keras import preprocessing
 
a = pd.read_csv('F:\\Study\\miniproject\\x_data\\1.csv', index_col=0, header =0, sep='\t')
b = pd.read_csv('F:\\Study\\miniproject\\x_data\\2.csv',index_col=0, header =0, sep='\t')

x_data = pd.concat([a, b], axis =0)

y_data = pd.read_csv('F:\\Study\\miniproject\y_data\\욕취합.csv',index_col=0, header =0, sep='\t')
x = x_data.loc[:,"댓글 내용"].values

#  형태소 분석
from eunjeon import Mecab
tagger = Mecab()
out = pd.DataFrame(data = None,index=None,columns=["댓글"])
# print(x)
# idx = 0
for i in x:
    tmp = pd.Series([tagger.morphs(i)],index=["댓글"])
    out = out.append(tmp,ignore_index= True)
    # idx += 1

print(out)

# word to vec
import urllib.request
from konlpy.tag import Okt
from gensim.models.word2vec import Word2Vec
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

out = out.loc[:,"댓글"].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
# out = out.dropna(how='any') # Null 값 
print(out)

print(type(out))


