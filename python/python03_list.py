#자료형

#1. 리스트--*******중요*******----------------------------------------

a = [1,2,3,4,5]
b = [1,2,3,'a','b'] #리스트 안에는 문자열,숫자열 가능/ numpy에는 한가지만 가능
print(b)

print(a[0]+ a[3])
# print(b[0] + b[3]) #TypeError: unsupported operand type(s) for +: 'int' and 'str'

print(type(a)) #<class 'list'>
print(a[-2])
print(a[1:3])

a = [1,2,3,['a','b', 'c']]
print(a[1])
print(a[-1]) #abc
print(a[-1][1]) #b / 마지막 리스트에서 1번 인덱스

#1-2. list slicing------------------------------------
a = [1,2,3,4,5]
print(a[:2]) #1,2

#1-3. list plus----------------------------------------

a = [1,2,3]
b = [4,5,6]
print( a + b ) #리스트뒤에 리스트 붙임/ numpy로는 [5,7,9]로 나옴
c = [7,8,9,10]
print(a+c) 

print(a * 3)#[1, 2, 3, 1, 2, 3, 1, 2, 3]

# print(a[2] + 'hi') #TypeError: unsupported operand type(s) for +: 'int' and 'str' > a[2]를 형변환, str으로 바꾼다
print(str(a[2]) + 'hi')

###문제###
f = '5'
# print(a[2]+f)
print(a[2] +int(f)) #정수화!!

'''
import numpy as np
x = np.array([range(1,101), range(311,411), range(100)]).T 
y = np.array([range(101,201), range(711,811), range(100)]).T #0~99

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split ( 
    #x, y, random_state=66, shuffle=True,#default option, 원래대로 하고 싶으면 shuffle false
    x, y, shuffle= False,
    
    train_size=0.8
    )
print(type(x_train)) #<class 'numpy.ndarray'> /sklearn에서 nump로 바꿔줌/numpy 속도가 엄청빠름/같은 타입만 쓸 수 있음

'''

#리스트 관련 함수
a = [1,2,3]
a.append(4) #덧붙이다 / range 등으로 데이타 가져올 때 씀 '.append'붙임
print(a)

# a = a.append(5)
# print(a) #문법 에러/ None

a.append(5) #이렇게만 써야함*************************잊으면 안된다!!
print(a)

a = [1, 3, 4, 2]
a.sort() #정렬 
print(a) #[1, 2, 3, 4]

a.reverse() #역순
print(a) #[4,3,2,1]

print(a.index(3)) #색인 /==a[3]
print(a.index(3))      # ==a[1]

a.insert(0, 7) #0번째 인덱스를 7으로 하여, [7,4,3,2,1]이 됨
print(a)

a.insert(3,3)
print(a) #[7, 4, 3, 3, 2, 1]

a.remove(7) #인자값 제거
print(a) #[4, 3, 3, 2, 1]

a.remove(3)
print(a) #[4, 3, 2, 1], 먼저 걸린 것 지워진다

