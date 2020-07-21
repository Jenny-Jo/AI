tup=(3,)
print(tup)

# tuple에 dic data 저장
d = dict(name='Jenny', mail='test@test.com',country='Korea')
for key, value in d.items():
    print(key + ' > ' + value)
    
# tuple을 사용한 가변 인수
def foo(*args):
    print(args[0])
    print(args[1])
    print(args[2])

foo('bar', 999, [1,2,3])
