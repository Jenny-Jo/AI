# 3의 배수 fizz
# 5의 배수 buzz
# 3,5의 배수 fizzbuzz
# 그 외 그냥 숫자
# 정수 30에 대해 판단
'''
a = 30
def fizzbuzz(x):
    if x % 3 == 0 and x % 5 ==0:
        print('fizzbuzz')
    elif x % 3 == 0 :
        print('fizz')
    elif x % 5 == 0 :
        print('buzz')
    else:
        print(x)

fizzbuzz(30)        
    '''

x = 30

if x % 15 == 0:
    print('fizzbuzz')
elif x % 3 == 0 :
    print('fizz')
elif x % 5 == 0 :
    print('buzz')
else:
    print(x)
    
x = input('숫자 입력해주세요')
x = int(x)
if x % 15 == 0:
    print('fizzbuzz')
elif x % 3 == 0 :
    print('fizz')
elif x % 5 == 0 :
    print('buzz')
else:
    print(x)