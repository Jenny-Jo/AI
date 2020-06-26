# 맞았습니다!!

a = int(input())
b = int(input())
c = int(input())
d = int(input())

x = (a+b+c+d)//60
y = (a+b+c+d)%60

print(x)
print(y)

# 맞은 답변 참고했음
temp = 0
for i in range(4):
    sec = int(input())
    temp += sec
     
print('%d\n%d'%(temp//60,temp%60))

