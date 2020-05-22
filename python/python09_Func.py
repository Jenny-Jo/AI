### 함수 만들기 ###
def sum1(a, b):             # sum이란 함수에 a, b 값의 변수를 넣겠다
    return a + b            # c + d 이렇게 없는거 쓰면 안됨

# print(sum(3, 4))

a = 1
b = 2
c = sum1(a, b)              # 위치만 맞으면 됨

print(c)                    # 3

## 곱,나눗셈,뺄셈 함수 만들기 ##
# mul1, div1,sub1

#곱셈
def mul1(a, b) :
    return a * b        

a = 1
b = 2
c = mul1(a, b)
print(c)                    # 2

#나눗셈

def div1 (a, b) :
    return a / b            

a = 1
b = 2
c = div1(a, b)
print('div:  ', c)         # div:   0.5

#뺄셈
def sub1 (a, b) :
    return a - b

c = sub1(a, b)
print('sub1 : ', c)        # sub1 :  -1

def sayYeah():
    return 'Hi'
aaa = sayYeah()
print(aaa)
                          # Hi  ???
# #매개변수 (parameter) a, b가 없는 함수

def sum1(a, b,c): 
    return a + b +c

a = 1
b = 2
c = 34
d = sum1(a, b, c)
print(d)                # 37
