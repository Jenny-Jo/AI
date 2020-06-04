import numpy as np 
from keras.models import Sequential
from keras.layers import Dense, LSTM

# 1. data
a = np.array (range(1,11))
size = 5

def split_x(seq, size) :
    aaa = []
    for i in range(len(seq) - size +1):
        subset = seq [i : (i+size)]
        # aaa.append([item for item in subset])
        aaa.append(subset)
# dataset = split_x(a,size)에서 함수가 재사용 되기 때문에 seq = a 가 됨
    print(type(aaa))
    return np.array(aaa)

dataset = split_x(a, size)
print ("==================")
print(dataset)

# <class 'list'>
# ==================
# [[ 1  2  3  4  5]
#  [ 2  3  4  5  6]
#  [ 3  4  5  6  7]
#  [ 4  5  6  7  8]
#  [ 5  6  7  8  9]
#  [ 6  7  8  9 10]] # (6,5)

