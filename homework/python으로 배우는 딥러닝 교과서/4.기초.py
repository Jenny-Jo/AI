print("Hello world")
print(3+8) #3+8 출력
print(3+6)
print("6-3") # str

print(18)
print(2+6)
print('2+6')


print(3+5)
print(3-5)
print(3*5)
print(3%5) # 나머지
print(3**5)# 제곱

# 변수 갱신
x = 5
x+= 1 # x = x + 1
print(x)

x = 5
x*=2 # x = x * 2
print(x)

n = 14
n *=5
print(n)

p = "seoul"
print( 'I am ' + 'from ' + p )


height = 177
type(height)

h = 1.7
w = 60
print(type(h))
print(type(w))
bmi = w/h**2
print(w/h**2)
print(type(w/h**2))
print("당신의 BMI는" + str(bmi) + "입니다") # 부동소수점형을 문자열로 변환>끼리끼리 같은 문자열끼리 있어야 함

greeting = " hello!!"
print(greeting *5)

n = '10'
print(n*3)

print(1+1 ==3) # False


n = 16
if n > 15:
    print('큰 숫자')

for i in range(10) :
    if i ==1:
        print("1 입니다")
    else :
        print("1이외의 것")

n =15
if n > 6 :
    print("큰 숫자")

n_1 = 14
n_2 = 28

print(n_1>8 and n_1<14) # False
print(not n_1 **2 < n_2*5) # True

n = 2021
if n % 400 !=0 and n % 100 ==0  :
    print(str(n) + "은 평년")
elif n % 4 ==0 :
    print(str(n) + "은 윤년")
else :
    print(str(n)+"평년")