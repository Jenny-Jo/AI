#range 함수(class)
a = range(10)
print(a) #range(0,10)

b = range(1, 11)
print(b)

for i in a:
    print(i) #0부터 9까지

for i in b:
    print(i) #1부터 10
 
print(type(a)) #<class 'range'>

sum = 0
for i in range(1,11): #1부터 10까지
    sum = sum + i 
print(sum) 