#range 함수(class)
a = range(10)
print(a)                            # range(0,10)

b = range(1, 11)
print(b)                            # range(1,11)

for i in a:
    print(i)                        # 0부터 9까지 출력

for i in b:
    print(i)                        # 1부터 10까지 출력
 
print(type(a))                      # <class 'range'>

sum = 0
for i in range(1,11):              
    sum = sum + i 
print(sum)                          # 1부터 10까지 다 더하여 55 나옴