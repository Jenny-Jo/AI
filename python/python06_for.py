
###### for 반복문 ######

a = {'name':'Jo', 'phone':'010', 'birth':'0509'} #문자형도 됨

for i in a.keys():                               # 다음줄로 엔터치면 자동으로 들여쓰기 해서 for 문 안에 들어있음을 명시
    print(i)                                     # 각 keys가 i에 들어가
                                                 # i 정의는 이미 되어있고, i 대신 다른 것 넣어줘도 됨, 그러나 통상적으로 i 씀
                   

                                                # name
                                                # phone
                                                # birth

#----------------------------------------------------------------------------
a = [1,2,3,4,5,6,7,8,9,10]
for i in a:                                     # a인자 갯수 10번만큼 돌려라/
    i = i*i
    print(i)                                    
    print('Hello')                              # Hello를 for문 안에서 10번 같이 출력 
print('Hello')                                  # 한번 출력/for문과 동등 

# 1
# Hello
# 4
# Hello
# 9
# Hello
# 16
# Hello
# 25
# Hello
# 36
# Hello
# 49
# Hello
# 64
# Hello
# 81
# Hello
# 100
# Hello
# Hello

#----------------------------------------------------------------------------

for i in a:
    print(i)
# 1
# 2
# 3
# 4
# 5
# 6
# 7
# 8
# 9
# 10

######## while문 #########

'''
while 조건문 :        #참 True인 동안 계속 돈다 ???
    수행할 문장
'''

######## if 문 #########
if 1 :
    print('True')
else :
    print('False')                                  # True 
                                                
 

if 3 :
    print('True')
else :
    print('False')                                  # Ture



if 0 :
    print('True')
else :
    print('False')                                  #False ?????



if -1:
    print('True')
else :
    print('False')                                  #True


'''
비교연산자

<,>, ==, !=, >=, <=

'''

# if a = 1:                                         # a에다 1을 넣는다 > 입력한다 라고 인식
#     print('출력될까')                              # SyntaxError: invalid syntax


a = 1
if a == 1:
    print('출력될까')
                                                    # 출력될까

money = 10000
if money >= 30000 :
    print('스파게티정식')
else:
    print('하겐다즈한통')
                                                    # 하겐다즈한통


#### 조건연산자 ####
# and, or, not

money = 20000
card = 1
if money >= 30000 or card ==1:                      # 둘 중 하나만 만족시키면 됨
    print('스파게티정식')
else :
    print ('하겐다즈한통')
                                                    # 스파게티정식

#--------------------------------------------
score = [90, 25, 67,45, 80]
number = 0
for i in score :                                    # i에 인수가 하나씩 들어간다 #i에 90들어가,
    if i >= 60:
        print("경]합격[축")
        number = number + 1
print ("합격인원:",number,"명")

                                                    # 경]합격[축
                                                    # 경]합격[축
                                                    # 경]합격[축
                                                    # 합격인원: 3 명

#------------------------------------------
#break, continue// break를 더 많이 쓴다


print("===============break=======================")

score = [90, 25, 67,45, 80]
number = 0
for i in score :                                    # i에 인수가 하나씩 들어간다 #i에 90들어가,
    if i < 30:
        break                                        # break 걸리면 제일 가까운 for문 중지시킴/25에 걸려서 멈춤
    
    if i >= 60:
        print("경]합격[축")
        number = number + 1
print ("합격인원:",number,"명")

                                                    # 경]합격[축
                                                    # 합격인원: 1 명

print("===============continue=======================")
score = [90, 25, 67,45, 80]
number = 0
for i in score : 
    if i < 60:
        continue 
    
    if i >= 60:
        print("경]합격[축")
        number = number + 1
print ("합격인원:",number,"명")

                                                    # 경]합격[축
                                                    # 경]합격[축
                                                    # 경]합격[축
                                                    # 합격인원: 3 명
