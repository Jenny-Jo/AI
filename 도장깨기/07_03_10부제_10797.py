'''
--url--
https://www.acmicpc.net/problem/10797

--title--
10797번: 10부제

--problem_description--
서울시는 6월 1일부터 교통 혼잡을 막기 위해서 자동차 10부제를 시행한다. 
자동차 10부제는 자동차 번호의 일의 자리 숫자와 날짜의 일의 자리 숫자가 일치하면 해당 자동차의 운행을 금지하는 것이다. 
예를 들어, 자동차 번호의 일의 자리 숫자가 7이면 7일, 17일, 27일에 운행하지 못한다.
 또한, 자동차 번호의 일의 자리 숫자가 0이면 10일, 20일, 30일에 운행하지 못한다.

여러분들은 일일 경찰관이 되어 10부제를 위반하는 자동차의 대수를 세는 봉사활동을 하려고 한다.
날짜의 일의 자리 숫자가 주어지고 5대의 자동차 번호의 일의 자리 숫자가 주어졌을 때 위반하는 자동차의 대수를 출력하면 된다. 

--problem_input--
첫 줄에는 날짜의 일의 자리 숫자가 주어지고 두 번째 줄에는 5대의 자동차 번호의 일의 자리 숫자가 주어진다. 
날짜와 자동차의 일의 자리 숫자는 모두 0에서 9까지의 정수 중 하나이다. 

--problem_output--
주어진 날짜와 자동차의 일의 자리 숫자를 보고 10부제를 위반하는 차량의 대수를 출력한다.

'''

# % 나머지

date = int(input())

cars = []
a = map(int, input().split())

violated_cars = []
for i in range(5):
    if cars[i] == date:
        violated_cars.append(cars[i])
    else:
        continue
print(len(violated_cars))
# 리스트로 한다면 어딜 고치면 좋을까요??? 고수님들...부탁드려요....

###
y = int(input())
a, b, c, d, e = map(int, input().split())
cars = (a, b, c, d, e)
count = 0
for x in cars:
    if x == y:
        count += 1
print(count)
###



