#2번의 첫번째 답--------<keras 49>-------------------------------
# x = [1, 2, 3] #리스트
# x = x - 1 
# print(x) #리스트로 하면 오류뜸/ 넘파이로 하면 수학적으로 인공지능이 이해함

'''
import numpy as np
y = np.array([1,2,3,4,5,1,2,3,4,5])
y = y - 1

print(y) #[0 1 2 3 4 0 1 2 3 4]

# from keras.utils import np_utils
# y = np_utils.to_categorical (y) #to_categorical  이용하기 위해 무조건적으로 0으로 시작
# print(y)
# print(y.shape)
'''

# 2번의 두번째 답 ---<keras 48>----------------------------------
import numpy as np

y = np.array([1,2,3,4,5,1,2,3,4,5])
print(y.shape) 
y = y.reshape(-1, 1) #one hot encoder은 2차원을 넣어줘야함//전체에서 일을 빼서???
# y = y.reshape(10,1) shape (10,6) 에서 (10,5)
print(y)
'''
[[1]
 [2]
 [3]
 [4]
 [5]
 [1]
 [2]
 [3]
 [4]
 [5]]
 '''
from sklearn.preprocessing import OneHotEncoder
aaa = OneHotEncoder (  )
aaa.fit(y)
y = aaa.transform(y).toarray() ##????
print(y)
print(y.shape)
