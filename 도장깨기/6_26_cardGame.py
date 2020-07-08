# JOI군은 카드 게임을 하고 있다. 이 카드 게임은 5회의 게임으로 진행되며, 그 총점으로 승부를 하는 게임이다.

# JOI군의 각 게임의 득점을 나타내는 정수가 주어졌을 때, JOI군의 총점을 구하는 프로그램을 작성하라.
# 
# 표준 입력에서 다음과 같은 데이터를 읽어온다.

#    i 번째 줄(1 ≤ i ≤ 5)에는 정수 Ai가 적혀있다. 이것은 i번째 게임에서의 JOI군의 점수를 나타낸다.
#   모든 입력 데이터는 다음 조건을 만족한다.

#   0 ≤ Ai ≤ 100．
# 표준 출력에 JOI군의 총점을 한 줄로 출력하라.

score = 0
for i in range(5):
    score_of_each_game = int(input()) # Ai = int(input())
    score += score_of_each_game
print(score)




'''
# 1차 시도 - 틀렸음
from random import randint
score = 0
for i in range(1,6) :
    Ai = randint(100)
    score += Ai
print(score)

# 2차 시도 - 런타임 에러

from random import randint
score = 0
for i in range(1,6) :
    Ai = randint(101) # 형식 틀림
    score += Ai
print(score)

# 3차 시도  - 틀렸음
from random import randint
score = 0
for i in range(1,6) :
    Ai = randint(1,101)
    score += Ai
print(score)
# 표준 입력, 표준 출력에서 망한건가?

# 4차 시도 - 틀렸음
data = int(input())
name =input('what is your name? : ')
print('hello, %s'%name)

from random import randint
score = 0
for i in range(1,6) :
    Ai = int(input('score'))
    score += Ai
print(score)
'''
# 5차 시도 - 해민이 도움으로 맞았습니다!!
score = 0
for i in range(1,6):
    Ai =int(input())
    score += Ai 
print(score)
