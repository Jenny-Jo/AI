import numpy as np
a = np.array([[1,2,3,4],[5,6,7,8]])
print(a.shape)
print(a.reshape(4,2))

arr = np.array([[1,2,3],[4,5,6],[7,8,9]])
print(arr[0,2])
print(arr[1,:2], arr[2,:2])
print(arr[1:, :2])

# axis
# 행 axis =1
# 열 axis =0
print(arr.sum())
print(arr.sum(axis=0))
print(arr.sum(axis=1))

# 팬시 인덱스
arr= np.arange(25).reshape(5,5)
print(arr[[3,2,0]])

# 전치 행렬
arr= np.arange(10).reshape(2,5)
print(arr.T)
print(np.transpose(arr))

# 정렬
arr = np.array([[8,4,2],[3,5,1]])
print(arr.argsort())          #정렬된 인덱스 반환하는 메서드
print(np.sort(arr))           #정렬 함수
arr.sort(1)                   #행 정렬하는 메서드, 출력
print(arr)


#행렬계산
import numpy as np 
arr = np.arrange(9).reshape(3,3)
print()

import numpy as np 
arr = np.arrange(9).reshape(3,3)
print(np.dot(arr, arr))
vec = arr.reshape(9)
print(np.linalg.norm(vex))


# 통계함수
import numpy as np
arr = np.arange(15).reshape(3,5)
#각 열의 평균을 출력
print(arr.mean(axis=0))
#변수 arr의 행 합계를 구하시오 
print(arr.sum(axis=1))
#최솟값
print(arr.min())
#각 열의 최댓값의 인덱스 번호
print(arr.argmax(axis=0))

#브로드캐스트
x= np.arange(6).reshape(2,3)
print(x+1)


import numpy as np

x= np.arange(15).reshape(3,5)

y= np.array([np.arange(5)])

z= x - y
print(x)
# [[ 0  1  2  3  4]
#  [ 5  6  7  8  9]
#  [10 11 12 13 14]]



#연습문제
import numpy as np
np.random.seed(100)

arr = np.random.randint(0,31,(5,3))
print(arr)

arr = arr.T
print(arr)

arr3 = arr[:, 1:4]
print(arr3)

arr3.sort(0)
print(arr3)

print(arr3.mean(axis=0))

#종합문제
import numpy as np

np.random.seed(0)

def make_image(m,n):
    image = np.random.randint(0,6,(m,n))

    return image
def change_little(matrix):
    shape = matrix.shape

    for i in range(shape[0]):
        for j in range(shape[1]):
            if np.random.randint(0,2)==1:
                matrix[i][j] = np.random.randint(0,6,1)

    return matrix
image1 = make_image(3,3)
print(image1)
print()

image2 = change_little(np.copy(image1))
print(image2)
print()
image3 = image2-image1
print(image3)
print()
image3 = np.abs(image3)
print(image3)
