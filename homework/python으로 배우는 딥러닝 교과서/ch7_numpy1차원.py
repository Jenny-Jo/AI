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

import numpy as np
arr = np.array([2,5,3,4,8])

print(arr+arr)
print(arr-arr)
print(arr**3)
print(1/arr)

arr = np.arange(10)
print(arr)
print(arr[3:6])
arr[3:6]=24
print(arr)

# 복사할 배열.copy()
arr1= np.array([1,2,3,4,5])
arr2= arr1.copy()
arr2[0] = 100
print(arr1)

# arr_NumPy = np.arrage(10)
# arr_Numpy_copy = arr_NumPy[:].copy() 
# copy 사용시 복사본 생성 되어 arr_NumPy_copy는 arr_NumPy에 영향 미치지 않음

# 부울 인덱스 참조
import numpy as np
s = np.array([2,3,4,5,6,7])
print(s[s%2==0])
print(s%2==0) # [ True False  True False  True False]

# 범용 함수
import numpy as np
arr = np.array([4,-9,16,-4,20])
arr_abs = np.abs(arr)
arr_e = np.exp(arr_abs)
print(arr_e)
arr_sqrt = np.sqrt(arr_abs)
print(arr_sqrt)

# 집합 함수
arr1 = [2,5,7,9,2]
arr2 = [2,4,7,8,3]
new_arr1 = np.unique(arr1)
print(new_arr1)                         # 중복제거
print(np.union1d(new_arr1,arr2))        # 합집합
print(np.intersect1d(new_arr1,arr2))    # 교집합
print(np.setdiff1d(new_arr1,arr2))      # 차집합

# 난수
from numpy.random import randint, rand
# np.random.randit으로 매번 안쳐도 됨
arr1 = randint(0,11,(5,2))              # 0에서 10 사이 5*2 행렬의 난수
print(arr1)
arr2 = rand(3)                          # 0~1사이 3개의 난수
print(arr2)


