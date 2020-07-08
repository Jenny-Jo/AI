from keras.preprocessing.text import Tokenizer

# 1. 인덱싱해주기
text = '나는 맛있는 밥을 먹었다'
# text2 = 'I had great dinner'
token = Tokenizer()
token.fit_on_texts([text])
print(token.word_index)
# {'나는': 1, '맛있는': 2, '밥을': 3, '먹었다': 4}

x = token.texts_to_sequences([text])
print(x)
# [[1, 2, 3, 4]]


# 2. 원핫인코딩
from keras.utils import to_categorical
word_size = len(token.word_index) + 1
x = to_categorical(x, num_classes=word_size)
print(x)
'''[[[0. 1. 0. 0. 0.]
  [0. 0. 1. 0. 0.]
  [0. 0. 0. 1. 0.]
  [0. 0. 0. 0. 1.]]]'''
#문제점은 커진다는 것>압축하자

# 3. 임베딩 / 시계열, 자연어처리에 많이 쓰임













