#2. 튜플
#리스트와 거의 같으나, 삭제와 수정이 안된다.
a = (1, 2, 3)
b = 1, 2, 3
print(type(a))
print(type(b)) #<class 'tuple'>

# a.remove(2) #AttributeError: 'tuple' object has no attribute 'remove' 속성에러
# print(a)

print(a + b)
print(a * 3)

#(1, 2, 3, 1, 2, 3)
#(1, 2, 3, 1, 2, 3, 1, 2, 3)

print(a - 3)