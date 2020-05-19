#print문과 format function-------------------
a = '사과'
b = '배'
c = '옥수수'

print('스벅프라푸치노를 대령하라')

print(a)
print(a, b)
print(a, b, c)


#.format(x) 옛날방식----------------------------
print("나는 {0}를 먹었다".format(a))
#0번째 인덱스를 넣겠다 (첫번째)
print("나는 {0}와 {1}을 먹었다".format(a, b))
print("I ate {0},{1} and {2}.".format(a, b, c))

print('I', 'ate', a, '.')
print('I', 'ate', a, 'and', b, '.')
print('I', 'ate', a,',', b, 'and', c, '.')
print('나는',a +'와', b+'를','먹었다.') #,대신 +로 대체하여 공백 없애기
print('나는',a ,'와 ', b,'를 ','먹었다.',sep='') #공백 없애기 ''간격을 없앰.
print('나는',a ,'와 ', b,'를 ','먹었다.',sep='#') ##를 문자열 사이마다 넣어라
                                                 #나는#사과#와 #배#를 #먹었다.



