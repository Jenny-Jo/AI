from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tqdm import keras

model = Sequential()
# model.add(Conv2D
#          (10,                       #그 레이어의 아웃풋이 10개!!!!! filter !!!
#          (2,2),                     #가로세로 2,2로 자름 !!! kernal_size=(2,2) or  2 가로세로 2로 자름
#                  # ? 같은 길이가 되도록 해줌
#          input_shape=(5,5,1) )    # 3차원 // 끝에 명암/ 1 흑백 (흰0검1) 아님 3 칼라//Batch size (height, width, channel) > 4차원
#          )                          # 이 첫번째 레이어는 Embedding 층임

# # x = (10000장, 10, 10, 1) inputshape 행,가로,세로, 색깔???
# #  4차원/ 만장의 그림이 가로세로 10이다
# #이미지가 커지면 (100,100,1) 이런 식으로 숫자 커짐




model.add(Conv2D(10, (2,2), input_shape=(10,10,1)))             # (9, 9, 10)
model.add(Conv2D(7, (3,3) ))                                    #(7, 7, 7)
model.add(Conv2D(5, (2,2), padding= 'same'))  #(7, 7, 5)
model.add(Conv2D(5, (2,2) ))                                    #(6, 6, 5)
# model.add(Conv2D(5, (2,2), strides=2))                        #(3, 3, 5)
# model.add(Conv2D(5, (2,2), strides=2, padding='same'))        #(3, 3, 5) stride 우선순위
model.add(MaxPooling2D(pool_size=2))
model.add(Flatten())                                            #쫙 편다/ Dense에 넣을 수 있는 형태로 바뀌어져//
# flatten_1 (Flatten)          (None, 45)                0///hyper parameter 의 일환이다
model.add(Dense(1))


model.summary()