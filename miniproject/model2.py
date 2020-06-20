#pip3 install tensorflow
#pip3 install keras

import pandas as pd
#
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
#
from keras.preprocessing.text import Tokenizer
#
from keras.preprocessing import sequence
#
from sklearn import model_selection
from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Dense
from keras import callbacks

df_train = pd.read_csv('https://raw.githubusercontent.com/ipcplusplus/toxic-comments-classification/master/train.csv')
x_train = df_train["comment_text"].fillna("_na_").values
categories = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y_train = df_train[categories].values
df_test = pd.read_csv('https://raw.githubusercontent.com/ipcplusplus/toxic-comments-classification/master/test.csv')
x_test = df_test["comment_text"].fillna("_na_").values

x_train = x_train[:100]
y_train = y_train[:100]
x_test = x_test[:100]

nltk.download('punkt')
nltk.download('stopwords')
#print(stopwords.fileids()) #['danish', 'dutch', 'english', 'finnish', 'french', 'german', 'hungarian', 'italian', 'kazakh', 'norwegian', 'portuguese', 'russian', 'spanish', 'swedish', 'turkish'] #불용어를 제공하는 국가의 언어
#print(stopwords.words('english')[:10]) #['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're"
stemmer = SnowballStemmer('english')
for i, document in enumerate(x_train):
    words = word_tokenize(document)
    #print(words) #
    clean_words = []
    for word in words:
        word = word.lower()
        if word not in stopwords.words('english'): #불용어 제거
            word = stemmer.stem(word) #어간 추출
            clean_words.append(word)
    #print(clean_words) #['봄', '신제품', '소식']
    x_train[i] = ' '.join(clean_words)
#print(x_train[:2]) #['free lotteri', 'free get free', 'free scholarship', 'free contact', 'award', 'ticket lotteri', 'ticket lotteri', 'ticket lotteri', 'ticket lotteri', 'ticket lotteri']

for i, document in enumerate(x_test):
    words = word_tokenize(document)
    #print(words) #
    clean_words = []
    for word in words:
        word = word.lower()
        if word not in stopwords.words('english'): #불용어 제거
            word = stemmer.stem(word) #어간 추출
            clean_words.append(word)
    #print(clean_words) #['봄', '신제품', '소식']
    x_test[i] = ' '.join(clean_words)
#print(x_test[:2]) #['free lotteri', 'free get free', 'free scholarship', 'free contact', 'award', 'ticket lotteri', 'ticket lotteri', 'ticket lotteri', 'ticket lotteri', 'ticket lotteri']

max_features = 20000
#tokenizer = Tokenizer() #Turning each text into either a sequence of integers (each integer being the index of a token based on word count)
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(x_train)
#print(tokenizer.word_index) #{'lotteri': 1, 'free': 2, 'ticket': 3, 'get': 4, 'scholarship': 5, 'contact': 6, 'award': 7}
x_train = tokenizer.texts_to_sequences(x_train)
x_test = tokenizer.texts_to_sequences(x_test)
#print(x_train[:2]) #[[2, 1], [2, 4, 2], [2, 5], [2, 6], [7], [3, 1], [3, 1], [3, 1], [3, 1], [3, 1]]
#print(x_test[:2]) #[[2, 1], [2, 4, 2], [2, 5], [2, 6], [7], [3, 1], [3, 1], [3, 1], [3, 1], [3, 1]]
document_max_words = 50
x_train = sequence.pad_sequences(x_train, maxlen=document_max_words)
x_test = sequence.pad_sequences(x_train, maxlen=document_max_words)
#print(x_train[:2]) 
'''
[[0 0 0 0 0 0 0 0 2 1]
 [0 0 0 0 0 0 0 2 4 2]
 [0 0 0 0 0 0 0 0 2 5]
 [0 0 0 0 0 0 0 0 2 6]
 [0 0 0 0 0 0 0 0 0 7]
 [0 0 0 0 0 0 0 0 3 1]
 [0 0 0 0 0 0 0 0 3 1]
 [0 0 0 0 0 0 0 0 3 1]
 [0 0 0 0 0 0 0 0 3 1]
 [0 0 0 0 0 0 0 0 3 1]]
'''
#print(x_test[:2]) 

##########

#x_train, x_test, y_train, y_test = model_selection.train_test_split(x_data, y_data, test_size=0.3)

model = Sequential()
#input_dim: 단어 사전의 크기를 말하며 총 max_features(15000)개의 단어 종류가 있다는 의미입니다. 이 값은 앞서 reuters.load_data() 함수의 num_words 인자값과 동일해야 합니다.
#input_length: 단어의 수 즉 문장의 길이를 나타냅니다. 임베딩 레이어의 출력 크기는 샘플 수 * output_dim * input_lenth가 됩니다. 임베딩 레이어 다음에 Flatten 레이어가 온다면 반드시 input_lenth를 지정해야 합니다. 플래튼 레이어인 경우 입력 크기가 알아야 이를 1차원으로 만들어서 Dense 레이어에 전달할 수 있기 때문입니다.
model.add(Embedding(input_dim=max_features, output_dim=128, input_length=document_max_words))
model.add(LSTM(units=512, input_shape=(5, 128))) 
model.add(Dense(units=6, activation='sigmoid')) 

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()
'''
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding_1 (Embedding)      (None, 5, 128)            1280      
_________________________________________________________________
lstm_1 (LSTM)                (None, 512)               1312768   
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 513       
=================================================================
Total params: 1,314,561
Trainable params: 1,314,561
Non-trainable params: 0
_________________________________________________________________
'''

history = model.fit(x_train, y_train, epochs=2, batch_size=1, shuffle=True, callbacks=[callbacks.EarlyStopping(patience=10, verbose=1)]) #조기 종료

'''
import matplotlib.pyplot as plt
plt.subplot(2,1,1) #2행(row) 1열(column)중 첫 번째 subplot #
#print(history.history.keys()) #dict_keys(['val_loss', 'val_acc', 'loss', 'acc'])
plt.plot(history.history['loss'], color='red', label='train loss')  
plt.plot(history.history['val_loss'], color='blue', label='validation loss')  
plt.title('loss per epoch')  
plt.ylabel('loss')  
plt.xlabel('epoch')  
plt.legend(loc='upper left')
plt.subplot(2,1,2) #2행(row) 1열(column)중 두 번째 subplot #
plt.plot(history.history['acc'], color='red', label='train accuracy')  
plt.plot(history.history['val_acc'], color='blue', label='validation accuracy')  
plt.title('accuracy per epoch')  
plt.ylabel('accuracy')  
plt.xlabel('epoch')  
plt.legend(loc='upper left')
plt.show()
'''

####################

print(x_test[:2])
'''
[[  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
    0 138   9  53 270 658 659 660  68   2 271 661 662 663 397 664 665 666
   15   2  54 186   6   4  87  10 667 668 669 670 671 672]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
    0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0 673
  187 674 675  10 139 398  19   6 676 677 678 679 680 108]]
'''
#
y_predict = model.predict(x_test[:2])
#y_predict = model.predict_proba(x_test[:2])
print(y_predict)
'''
[[0.12459078 0.00636867 0.01645073 0.00226837 0.00704876 0.00200066]
 [0.12459517 0.00636795 0.01645111 0.00226865 0.00704842 0.00200077]]
'''
y_predict = model.predict_classes(x_test[:2])
print(y_predict)
'''
[0 0]
'''

y_predict = model.predict(x_test)
df_submission1 = pd.DataFrame(df_test["id"], columns=['id'])
df_submission2 = pd.DataFrame(y_predict, columns=categories)
df_submission3 = pd.concat([df_submission1,df_submission2],axis=1)
df_submission3.to_csv("kaggle_toxic_submission.csv",index=False)