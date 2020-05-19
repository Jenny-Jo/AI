#정수형###############################
a = 1
b = 2
c = a + b
print(c)

d = a * b
print (d)

e = a / b
print(e) #0.5/ 인간 친화적인 언어라 정수 나누기 정수하면 실수가 나올 수 있다

#실수형################################
a = 1.1
b = 2.2
c = a+b
print(c)
# 3.3000000000000003 / rounding error

d = a*b
print(d)
# 2.4200000000000004

e =a/b
print(e)
# 0.5


#문자형################################
a = 'hel'
b = 'lo'
c = a + b
print(c)


#-----------형변환-----------------
# 문자 + 숫자
a = 123 # int 정수
b = '45' # 따옴표는 문자취급, str
# c = a+b #Type Error
# print(c) 
#10000

#숫자를 문자변환 + 문자
a = 123
a = str(a)
b = '45'
c = a + b
print(c) #12345


#숫자로 바꾸기
a = 123
b = '45'
b = int(b)
c = a+b
print(c)
#------------------------------------

#문자열 : 연산하기
a = 'abcdefgh'
print(a[0]) #시작은 0, 뒤로 가면 -로 간다
print(a[3])
print(a[5])
print(a[-1])
print(a[-2])
print(type(a)) #<class 'str'> 문자열####자주씀###타입 알면 에러 뜨는 것을 막는 데 도움이 된다

b = 'xyz'
print( a + b ) #abcdefghxyz


#문자열 인덱싱/ 잘 틀리니까 유의하자!
a = 'Hello, Deep Learning' #띄어쓰기, 쉼표도 문자다!
print(a[7])   
print(a[-1])
print(a[-2])
print(a[3:9])
print(a[3:-5]) #-5는 미만처리! 빼기 하나
print(a[:-1]) #시작 index는 0
print(a[:1]) #다시!
print(a[1:])
print(a[5:-4])

'''
D
g
n
lo, De
lo, Deep Lea
Hello, Deep Learnin
H
ello, Deep Learning
, Deep Lear
'''
