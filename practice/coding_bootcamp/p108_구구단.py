a = [1,2,3,4,5,6,7,8,9]
aa = []

for i in a :
    for j in a :
        print(i*j, end = ' ')
    print('a ')

for i in range(101):
    print(i, end=' ')

for i in range(0,30,3):
    print(i)
    
print(pow(2,2,3)) # 2의 2 제곱, 3으로 나눔

def max(a,b):
    if a > b :
        return a
    return b

result = max(10,30)
print(result)