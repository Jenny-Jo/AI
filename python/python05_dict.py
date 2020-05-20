#3. Dictionary #중복 x
#{key : value} 쌍으로 움직인다
#key = index

a = {1 : 'hi', 2:'hello'}
print(a) #{1: 'hi'}
print(a[1]) #hi #key 자체를 index로 본다

b = {'hi' :1, 'hello' : 2}
print(b['hello']) #2

# Dictionary attribute removal/요소 삭제
del a[1] #delete
print(a) #{2: 'hello'}
del a[2]
print(a) #{}

a = {1 :'a', 1:'b', 1:'c'}
print(a) #{1: 'c'}

b = {1:'a', 2:'a', 3:'a'}
print(b) #{1: 'a', 2: 'a', 3: 'a'} 영희,철수,지민이 똑같이 백점 맞았다

a = {'name':'Jo', 'phone':'010', 'birth':'0509'} #문자형도 됨
print(a.keys())
print(a.values())
print(type(a))
print(a.get('name'))
print(a['name'])


# dict_keys(['name', 'phone', 'birth'])
# dict_values(['Jo', '010', '0509'])
# <class 'dict'>
# Jo
# Jo

#조건문과 반복문하면 개발가능
#컴퓨터는 단순연산,더하기만 반복(뺄셈,곱,나누기도 다 더하기로 함)