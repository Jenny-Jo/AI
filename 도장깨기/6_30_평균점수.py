# 10039번

"""
--url--
https://www.acmicpc.net/problem/10039

--title--
10039번: 평균 점수

--problem_description--
상현이가 가르치는 아이폰 앱 개발 수업의 수강생은 원섭, 세희, 상근, 숭, 강수이다.

어제 이 수업의 기말고사가 있었고, 상현이는 지금 학생들의 기말고사 시험지를 채점하고 있다. 기말고사 점수가 40점 이상인 학생들은 그 점수 그대로 자신의 성적이 된다. 하지만, 40점 미만인 학생들은 보충학습을 듣는 조건을 수락하면 40점을 받게 된다. 보충학습은 거부할 수 없기 때문에, 40점 미만인 학생들은 항상 40점을 받게 된다.

학생 5명의 점수가 주어졌을 때, 평균 점수를 구하는 프로그램을 작성하시오.

--problem_input--
입력은 총 5줄로 이루어져 있고, 원섭이의 점수, 세희의 점수, 상근이의 점수, 숭이의 점수, 강수의 점수가 순서대로 주어진다.

점수는 모두 0점 이상, 100점 이하인 5의 배수이다. 따라서, 평균 점수는 항상 정수이다. 

--problem_output--
첫째 줄에 학생 5명의 평균 점수를 출력한다.
"""


# 1차 시도 - 런타임 에러

import sys
ws, sh, sg, s, gs = map(int, sys.stdin.readline().split())
# ws, sh, sg, s, gs = map(input().split())

print(int((ws+sh+sg+s+gs)/5))

# 2차 시도 - 런타임 에러

# import sys
# ws, sh, sg, s, gs = map(sys.stdin.readline().split())
ws, sh, sg, s, gs = map(int, input().split())
mean = int((ws+sh+sg+s+gs)/5)
print(mean)

# 3차 시도  - 런타임 에러
# 조건을 넣어봐야겠다
ws, sh, sg, s, gs = map(int, input().split())
[ws, sh, sg, s, gs]%5==0
40 <[ws, sh,sg, s, gs]<=100
mean = int((ws+sh+sg+s+gs)/5)
print(mean)

# N차 시도 - 구글링! -런타임에러
ws, sh, sg, s, gs = map(int, input().split())
if [ws, sh, sg, s, gs]%5==0:
    if 40 <[ws, sh,sg, s, gs]<=100:
        score = { 'ws': ws, 'sh': sh, 'sg':sg, 's':s, 'gs':gs}
        average = sum(score.values())/len(score)
        
# N차 시도 - 조건을 뺌 - 포기....
ws, sh, sg, s, gs = map(int, input().split())
score = { 'ws': ws, 'sh': sh, 'sg':sg, 's':s, 'gs':gs}
average = sum(score.values())/len(score)
print(average)

# 정답 봄- int 안해주면 자꾸 틀리더라...
sum = 0
for i in range(5):
    score = input()
    if int(score)< 40:
        score = 40
    sum += int(score)
avg = sum/5
print(int(avg))

