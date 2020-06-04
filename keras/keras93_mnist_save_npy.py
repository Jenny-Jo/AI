import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist                         
mnist.load_data()                                         

(x_train, y_train), (x_test, y_test) = mnist.load_data() 
print(x_train[0])                                         
print('y_train: ' , y_train[0])                           # 5

print(x_train.shape)                                      # (60000, 28, 28)
print(x_test.shape)                                       # (10000, 28, 28)
print(y_train.shape)                                      # (60000,)       
print(y_test.shape)                                       # (10000,)


np.save('./data/mnist_train_x.npy', arr=x_train)
np.save('./data/mnist_test_x.npy', arr=x_test)
np.save('./data/mnist_train_y.npy', arr=y_train)
np.save('./data/mnist_test_y.npy', arr=y_test)


# 데이터 전처리 1. 원핫인코딩 : 당연하다             
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train.shape)                                      #  (60000, 10)

# 데이터 전처리 2. 정규화( MinMaxScalar )                                                    
x_train = x_train.reshape(60000, 28, 28, 1).astype('float32') /255  
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32') /255.                                     

'''
1 ) 데이타 save
전처리 전까지를 save하고/ load하기

from sklearn.datasets import mnist
import numpy as np
#넘파이는 한가지 자료만 쓸 수 있다
#자료형에 유연한건 판다스
iris = load_iris()

print(type(iris)) #<class 'sklearn.utils.Bunch'>

x_data = iris.data
y_data = iris.target

print(x_data)
print(y_data)

np.save('./data/iris_x.npy', arr=x_data)
np.save('./data/iris_y.npy', arr=y_data)
x_data_load = np.load('./data/iris_x.npy')
y_data_load = np.load('./data/iris_y.npy')

print(type(x_data_load)) #<class 'numpy.ndarray'>
print(type(y_data_load))
print(x_data_load.shape) #(150,4)
print(y_data_load.shape) #(150,)

--------------------------------------------------------
np.save('./data/mnist_train_x.npy', arr=x_train)
np.save('./data/mnist_test_x.npy', arr=x_test)
np.save('./data/mnist_train_y.npy', arr=y_train)
np.save('./data/mnist_test_y.npy', arr=y_test)
-------------------------------------------------------
'''