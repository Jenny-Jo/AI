# 122 copy
# embedding 빼고 lstm으로 완성

from keras.preprocessing.text import Tokenizer
import numpy as np

docs = ['너무 재미있어요','참 최고에요','참 잘 만든 영화에요','추천하고 싶은 영화입니다', '한 번 더 보고 싶네요','글쎄요','별로에요','생각보다 지루해요','연기가 어색해요',
        '재미없어요','너무 재미없다','참 재밌네요']

# 긍정 1, 부정 0
labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1])

# 토큰화
token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)

# 중복된 '너무','참' 빼고 인덱스만 나와/ 제일 많이 나온 애들을 1로 index 줌
# {'참': 1, '너무': 2, '재미있어요': 3, '최고에요': 4, '잘': 5, '만든': 6, '영화에요': 7, '추천하고': 8, '싶은': 9, '영화입니다': 10, '한번': 11, '더': 12, '보고': 13, '싶네요': 14, '글쎄요': 15, '별로에요': 16, '생각보다': 17, '지루해요': 18, '연기가': 19, '어색해요': 20, '재미없어요': 21, '재미없다': 22, '재밌네요': 23}
# 단어별로  index 준 것

x = token.texts_to_sequences(docs)
print(x)
# [[2, 3], [1, 4], [1, 5, 6, 7], [8, 9, 10], [11, 12, 13, 14], [15], [16], [17, 18], [19, 20], [21], [2, 22], [1, 23]]
# 단어를 숫자로 변환시켜서 나열
# shape 이 다 다름 > 맞춰줘야 한다 > padding 0 으로 채워줌
from keras.preprocessing.sequence import pad_sequences
pad_x = pad_sequences(x, padding='post', value=1.0) # pre, post 도 있다 # padding, value default로 pre, 0으로 인식
print(pad_x)  #(12,5)
# padding = pre > 0 이 앞에서 채워줘 / value =1 하면 0대신 1로 채워줌
'''
[[ 2  3  1  1  1]
 [ 1  4  1  1  1]
 [ 1  5  6  7  1]
 [ 8  9 10  1  1]
 [11 12 13 14 15]
 [16  1  1  1  1]
 [17  1  1  1  1]
 [18 19  1  1  1]
 [20 21  1  1  1]
 [22  1  1  1  1]
 [ 2 23  1  1  1]
 [ 1 24  1  1  1]]'''

 # 참 쉽죠?
 
 # data ready > modling with imbedding for the next time!!
 
# 단어별 유사점을 벡터화한다

word_size = len(token.word_index) + 1
print('전체 토큰 사이즈: ', word_size ) # 전체 토큰 사이즈:  25

pad_x = pad_x.reshape(12,5,1) ############

from keras.models import Sequential
from keras.layers import Dense, Embedding, Flatten
from keras.layers.recurrent import LSTM

model = Sequential()
# model.add(Embedding(word_size, 10, input_length = 5)) # word_size(input1) 25 / output 10 / input_length(input2) 
# model.add(Embedding(25, 10, input_length=5)) # word_size(input1) 25 / output 10 / input_length(input2) 
# size 달라도 돌아간다
# model.add(Embedding(25,10)) #3차원 내뱉고
model.add(LSTM(3, input_shape = (5,1) )) #3차원 들어가
# model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

model.summary()
# 3차원 lstm, conv1d

# 임베딩(넌, 5, 10)
# 이중암시...밑밥 이미 깔렸음
# X, Y  넣을 때 바꿀 수 있다면?

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['acc'])
model.fit(pad_x, labels, epochs=30)
acc = model.evaluate(pad_x, labels)[1] # 0이 뭐였지? loss??
print(acc)


