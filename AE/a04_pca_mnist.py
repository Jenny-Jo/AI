# x를 4차원에서 2차원으로 변형, Dense 모델에 넣어주기
# keras 56_mnist_DNN.py 복붙


from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
import numpy as np
import matplotlib.pyplot as plt

#Datasets 불러오기
from tensorflow.keras.datasets import mnist  

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train[0])


print('y_train : ' , y_train[0])


print(x_train.shape)                   #(60000, 28, 28)
print(x_test.shape)                    #(10000, 28, 28)
print(y_train.shape)                   #(60000,) 스칼라, 1 dim(vector)
print(y_test.shape)                    #(10000,)


print(x_train[0].shape) #(28,28) 짜리 
# plt.imshow(x_train[0], 'gray')
# plt.imshow(x_train[0])
# plt.show()


# 2. x-------------------------------------------------------------------

# Data 전처리/ 2. 정규화 x
# 형을 실수형으로 변환
# # MinMax scaler (x - 최대)/ (최대 - 최소)

############ 4차원을 2차원으로#########
x_train = x_train.reshape(60000, 784).astype('float32')/255 ##
x_test  = x_test.reshape (10000, 784).astype('float32')/255 ##
######################################

X = np.append(x_train, x_test, axis=0)
print(X.shape)
# (70000, 784)

from sklearn.decomposition import PCA

pca = PCA()
pca.fit(X)
cumsum = np.cumsum(pca.explained_variance_ratio_)
print(cumsum)
n_components = np.argmax(cumsum >= 0.95 )+1
print(n_components) # 154