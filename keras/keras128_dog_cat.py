# 개, 고양이 분류
# 오라오라왈ㄹ와로아롸오라 냐냐야냐야냥냥냥오오옹 ㅇ라ㅗ아롸오라왈왈 냐안얀아냥앙

from keras.applications.vgg16 import VGG16
import matplotlib.pyplot as plt


from keras.preprocessing.image import load_img

img_dog = load_img('F:/Study/data/dog_cat/dog.jpg', target_size=(224, 224))
img_cat = load_img('F:/Study/data/dog_cat/cat.jpg', target_size=(224, 224))
img_suit = load_img('F:/Study/data/dog_cat/suit.jpg', target_size=(224, 224))
img_yang = load_img('F:/Study/data/dog_cat/yang.jpg', target_size=(224, 224))
# img_lana = load_img('F:/Study/data/dog_cat/lana.jpg', target_size=(224, 224))

plt.imshow(img_yang)
# plt.imshow(img_lana)

# plt.show()
              
from keras.preprocessing.image import img_to_array

arr_dog = img_to_array(img_dog)     
arr_cat = img_to_array(img_cat)     
arr_suit = img_to_array(img_suit)     
arr_yang = img_to_array(img_yang)     
print(arr_dog)
print(type(arr_dog)) # <class 'numpy.ndarray'> 어떤 파일이든 넘파이로 바꿀 수 있음 됨
print(arr_dog.shape) # (224, 224, 3)

# RGB > BGR : standard scaler형식
from keras.applications.vgg16 import preprocess_input
arr_cat = preprocess_input(arr_cat)
arr_dog = preprocess_input(arr_dog)
arr_suit = preprocess_input(arr_suit)
arr_yang = preprocess_input(arr_yang)

print(arr_dog)
print(arr_dog.shape
      )
# image data를 하나로
import numpy as np
arr_input = np.stack([arr_dog, arr_cat, arr_suit, arr_yang])

print(arr_input.shape) # (4, 224, 224, 3)

# 2. 모델 구성

model = VGG16()
probs = model.predict(arr_input)
print('probs.shape:', probs.shape) # probs.shape: (4, 1000)

# 3. 이미지 결과
from keras.applications.vgg16 import decode_predictions

results = decode_predictions(probs)
print('--------------------------------')
print(results[0])
print('--------------------------------')
print(results[1])
print('--------------------------------')
print(results[2])
print('--------------------------------')
print(results[3])
'''천개정도 판별 가능
--------------------------------
[('n02099601', 'golden_retriever', 0.4905493), ('n02099712', 'Labrador_retriever', 0.35917053), ('n02088466', 'bloodhound', 0.05587562), ('n02108551', 'Tibetan_mastiff', 0.036730338), ('n02109525', 'Saint_Bernard', 0.017359532)]
--------------------------------
[('n02123159', 'tiger_cat', 0.7365471), ('n02124075', 'Egyptian_cat', 0.17492676), ('n02123045', 'tabby', 0.04588403), ('n02127052', 'lynx', 0.015686696), ('n02129604', 'tiger', 0.0023856775)]
--------------------------------
[('n04350905', 'suit', 0.8674235), ('n04591157', 'Windsor_tie', 0.09708722), ('n02883205', 'bow_tie', 0.023431225), ('n10148035', 'groom', 0.0051312037), ('n03680355', 'Loafer', 0.004053905)]
--------------------------------
[('n04584207', 'wig', 0.14707443), ('n03000247', 'chain_mail', 0.14465734), ('n03877472', 'pajama', 0.07643354), ('n03450230', 'gown', 0.065038525), ('n03710637', 'maillot', 0.057375774)]'''