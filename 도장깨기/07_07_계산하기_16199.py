'''
--url--
https://www.acmicpc.net/problem/16199

--title--
16199번: 나이 계산하기

--problem_description--
한국에서 나이는 총 3가지 종류가 있다.

만 나이는 생일을 기준으로 계산한다. 어떤 사람이 태어났을 때, 그 사람의 나이는 0세이고, 생일이 지날 때마다 1세가 증가한다. 
예를 들어, 생일이 2003년 3월 5일인 사람은 2004년 3월 4일까지 0세이고, 2004년 3월 5일부터 2005년 3월 4일까지 1세이다.

세는 나이는 생년을 기준으로 계산한다. 어떤 사람이 태어났을 때, 그 사람의 나이는 1세이고, 연도가 바뀔 때마다 1세가 증가한다.
 예를 들어, 생일이 2003년 3월 5일인 사람은 2003년 12월 31일까지 1세이고, 2004년 1월 1일부터 2004년 12월 31일까지 2세이다.

연 나이는 생년을 기준으로 계산하고, 현재 연도에서 생년을 뺀 값이다. 
예를 들어, 생일이 2003년 3월 5일인 사람은 2003년 12월 31일까지 0세이고, 2004년 1월 1일부터 2004년 12월 31일까지 1세이다.

어떤 사람의 생년월일과 기준 날짜가 주어졌을 때, 기준 날짜를 기준으로 그 사람의 만 나이, 세는 나이, 연 나이를 모두 구하는 프로그램을 작성하시오.

--problem_input--
첫째 줄에 어떤 사람이 태어난 연도, 월, 일이 주어진다. 생년월일은 공백으로 구분되어져 있고, 항상 올바른 날짜만 주어진다.

둘째 줄에 기준 날짜가 주어진다. 기준 날짜도 공백으로 구분되어져 있으며, 올바른 날짜만 주어진다.

입력으로 주어지는 생년월일은 기준 날짜와 같거나 그 이전이다.

입력으로 주어지는 연도는 1900년보다 크거나 같고, 2100년보다 작거나 같다.

--problem_output--
첫째 줄에 만 나이, 둘째 줄에 세는 나이, 셋째 줄에 연 나이를 출력한다.

'''
man_age=0
count_age=0
year_age=0
​
birthday = list(map(int,input().split()))
today = list(map(int,input().split()))
​
if today[1]>birthday[1] or (today[1]==birthday[1] and today[2]>=birthday[2]):
    man_age=today[0]-birthday[0]
else:
    man_age=today[0]-birthday[0]-1
count_age=today[0]-birthday[0]+1
year_age=today[0]-birthday[0]
print(man_age)
print(count_age)
print(year_age)


​
a, b, c = map(int, sys.stdin.readline().split())
x, y, z = map(int, sys.stdin.readline().split())
​
manyear = x-a
month = y-b
day = z-c
if day<0:
    month -= 1
​
if month <0:
    manyear -= 1
​
yeonyear = x-a 
​
​
print(manyear)
print(yeonyear+1)
print(yeonyear)
Collapse



:100:
2


