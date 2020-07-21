import random

# size = random.randint(1,100)

size = int(input('size는?'))
list1 = []
def make_random_list1(size):
    for i in range(size):
        x = random.randint(0,100)
        list1.append(x)
    return list1
x = make_random_list1(size)

print(x)
y = x.sort() # none으로 뜨는 이유??
print(y)
