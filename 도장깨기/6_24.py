# ---[1271] 엄청난부자---틀림-----------------------------------

a = divmod(int(n), int(m))
print(a[0])
print(a[1])

# n과 m을 정수형으로 만들고 n/m하여 나온 값으로 a = (몫, 나머지)가 나올때
# a[0]은 몫, a[1]은 나머지
n, m = map(int, input().split())
print(n//m)
print(n%m)



# ----[8393] 합---------틀림----------------------------

a = 0
n = int(input())
for i in range(n):
    a += i +1
print(a)

# 1에서 n번째까지 for문을 돌려서 a = 0에다 순차적으로 더함
# range 안에다가 int(input())넣는 건 안되나 보다



n = range(1,5)
a = 0
for i in n:
    a = i + a
print(a)

# ------[1000번] 번외--------틀림---------------------------

a, b = map(int, input().split())
print(a+b)
# 틀려서 이미 답을 보았고, map과 split 함수까지 어떻게 이해를 해보려고 했으나 잘 이해가 안감



