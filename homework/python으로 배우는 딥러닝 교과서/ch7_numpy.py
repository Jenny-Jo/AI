#고속 처리 경험

import numpy as np
import time
from numpy.random import rand

N = 150
matA = np.array(rand(N,N))
matB = np.array(rand(N,N))
matC = np.array([[0]*N for _ in range(N)])

start = time.time()

for i in range(N):
    for j in range(N):
        for k in range(N):
            matC[i][j] = matA[i][k] * matB[k][j]

print("파이썬 기능만으로 계산한 결과 : %.2f[sec]"%float(time.time() - start))
start = time.time()
matC = np.dot(matA, matB)
print("Numpy 를 사용하여 계산한 결과 : %.2f[sec]" % float(time.time() - start))

# 1차원 배열 np.array
import numpy as np
storages = [24,3,54,6,7,12]
print(storages)
np_storages = np.array(storages)
print(type(np_storages))
