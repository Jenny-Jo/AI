# 1330
# 문제
# 두 정수 A와 B가 주어졌을 때, A와 B를 비교하는 프로그램을 작성하시오.

# 입력
# 첫째 줄에 A와 B가 주어진다. A와 B는 공백 한 칸으로 구분되어져 있다.

# 출력
# 첫째 줄에 다음 세 가지 중 하나를 출력한다.

# A가 B보다 큰 경우에는 '>'를 출력한다.
# A가 B보다 작은 경우에는 '<'를 출력한다.
# A와 B가 같은 경우에는 '=='를 출력한다.


# 1차 시도 - 컴파일 에러
A, B = map(int, input().split())

if A > B :
    print( " > ")
elif A == B :
    print(" == ")
else  A < B :
    print(" < ")
# else는 아예 조건 안쓰는 건데 모르고 씀. 구글링하고 알게 됨

# 2차 시도 - 런타임 에러

A, B = map(int, input().split())

if A > B :
    print( " > ")
elif A == B :
    print(" == ")
elif  A < B :
    print(" < ")
    

# 3차 시도 - 출력 형식 잘못되었습니다
A, B = map(int, input().split())

if A > B :
    print( " > ")
elif A == B :
    print(" == ")
else:
    print(" < ")


# N차 시도- 맞았습니다!!

A, B = map(int, input().split())

if A > B :
    print(' > ')
elif A < B:
    print(' < ')
else :
    print(' == ')

# "  " 에서  '  ' 로 바꿨는데 맞았다. 그 이유가 뭘까???