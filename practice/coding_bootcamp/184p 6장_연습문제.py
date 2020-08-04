def test(idx):
    try:
        abc = ['a','b','c']
        print(abc[idx])
    except IndexError as ie:
        print('인덱스가 범위를 벗어났습니다')

idx = int(input())
test(idx)
# 함수 안에 try, except 적어놓음


def add_number(d,e):
    try:
        return d + e
    except TypeError as TE:
        print('숫자가 아닌 데이터를 지정했습니다')

d = input('d : ')
e = input('e : ')
# input으로 들어가면 문자열로 들어간다!!
x = add_number(d,e)
print(x)
# add_number(10,'abc')
# x = add_number(10,10)
# print(x)