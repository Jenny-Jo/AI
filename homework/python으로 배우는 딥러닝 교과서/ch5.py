# 1. list형
c = ['red', 'blue', 'yellow']
print(c)

n = 3
print('apple', n)

apple = 4
grape = 3
banana = 6
fruits = [apple, grape, banana]
print(fruits)

c = ['dog', 'blue', 'yellow']
c[0] = 'red'
c  += ['green'] # c.append('green')
print(c)

c = ['dog', 'blue', 'yellow']
del c[0]
print(c)

c = ['red', 'blue', 'yellow']
c_copy = c[:] # c_copy = list(c)
c_copy[1]='green'
print(c)
 
# 2. dictonary
town = {'경기도':'수원', '서울':'중구'}
print(town)
print(type(town))
print(" 경기도의 중심지는" + town['경기도'] + "입니다")
print(" 서울의 중심지는" + town['서울'] + "입니다")
town['제주도']='제주시'
town['경기도']='분당'
print(town)
del town['경기도']
print(town)

# 3. while문

x=5
while x > 0:
    print("hanbit")
    x-=2
#???
x = 5
while x != 0 :
    x-=1
    if x !=0:
        print(x)
    else:
        print('bang')
print('===========4===============')
# 4. for 문
storages = [1,2,3,4]
for i in storages:
    print(i)

# alphabet = ['a','b','c','d','e','f','g']
number = [1,2,3,4,5,6]
for i in number:
    print(i)
    if i>=5:
        print('end')
        break

for i in number:
    print(i)
    if i == 4:
        break
print('------------')
number = [1,2,3,4,5,6]
for i in number:
    if i % 2 == 0:
        continue
    print(i)
print('------------')

a = ['a','b']
for c, d in enumerate(a):
    print(c, d)

animals = ['tiger', 'dog','elephant']
for a, b in enumerate(animals):
    print('index:'+str(a ) , b)
    # print('index:'+str(a ) + b) +와 ,의 차이는? + 는 바로 붙여 쓰지만 ,는 띄어쓰기 있음

fruits = [['strawberry', 'red'],['peach','pink'],['banana','yellow']]
for a, b in fruits:
    print(a,"is",b)


