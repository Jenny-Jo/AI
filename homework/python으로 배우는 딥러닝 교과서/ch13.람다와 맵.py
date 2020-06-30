# 13.1 람다식의 기초
# 13.1.1 람다함수 -코드 절약!
# def 보다 더 간단하게 코드 작성 가능
a = 4
def func1(a):
    return 2*a**2 -3*a + 1 
print(func1(4))

func2 = lambda a : 2*a**2 - 3*a + 1
print(func2(a))

# 13.1.2 다변수 함수
x = 5
y = 6
z = 2

def func3(x,y,z) :
    return x*y + z

func4 = lambda x,y,z : x*y + z

print(func3(x,y,z))
print(func4(x,y,z))

# 13.1.3 if 이용 - 삼항연산자 표기
# 람다는 반환값에 식만 가능. 문자열은 안됨
a1 = 13
a2 = 32
                               # 10=<a<30 이건 안됨  # a = 50 이것도 아님
func5 = lambda a:a**2-40*a+350 if a>=10 and a <30 else 50
print(func5(a1))
print(func5(a2))

# 13.2 편리한 표기법
# 13.2.1 하나의 기호로 리스트 분할
self_data = "My name is Jenny Jo"
splited = self_data.split(" ")
print(splited)
print(splited[3])
# 나눌문자열.split("구분기호", 분할횟수)

# 13.2.2 여러 기호로 리스트 분할
import re
test_sentence = "this, is a.test,sentence"
re.split("[,.]", test_sentence)

import re
time_data = "2020/06/30_12:47"
splited_time_data = re.split("[/_:]", time_data)
print(splited_time_data[1], splited_time_data[3])
