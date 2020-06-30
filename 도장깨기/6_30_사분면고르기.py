# 14681번
"""
--url--
https://www.acmicpc.net/problem/14681

--title--
14681번: 사분면 고르기

--problem_description--
흔한 수학 문제 중 하나는 주어진 점이 어느 사분면에 속하는지 알아내는 것이다. 사분면은 아래 그림처럼 1부터 4까지 번호를 갖는다. "Quadrant n"은 "제n사분면"이라는 뜻이다.

None

예를 들어, 좌표가 (12, 5)인 점 A는 x좌표와 y좌표가 모두 양수이므로 제1사분면에 속한다. 점 B는 x좌표가 음수이고 y좌표가 양수이므로 제2사분면에 속한다.

점의 좌표를 입력받아 그 점이 어느 사분면에 속하는지 알아내는 프로그램을 작성하시오. 단, x좌표와 y좌표는 모두 양수나 음수라고 가정한다.

--problem_input--
첫 줄에는 정수 x가 주어진다. (−1000 ≤ x ≤ 1000; x ≠ 0) 다음 줄에는 정수 y가 주어진다. (−1000 ≤ y ≤ 1000; y ≠ 0)

--problem_output--
점 (x, y)의 사분면 번호(1, 2, 3, 4 중 하나)를 출력한다."""

# 1차 시도 - 런타임 에러
# 첫줄, 다음줄 따로 입력하는거라 map()함수 안썼다
import sys
x = int(sys.stdin())
y = int(sys.stdin())

if x>0:
    if y>0:
        print(1)
    if y<0:
        print(4)
if x<0:
    if y>0:
        print(2)
    if y<0:
        print(3)

# 2차 시도 -맞았습니다!! - sys 에서 문제가 일어난건가 해서 input()으로 바꿈
x = int(input())
y = int(input())

if x>0:
    if y>0:
        print(1)
    if y<0:
        print(4)
if x<0:
    if y>0:
        print(2)
    if y<0:
        print(3)

# 3차 시도 - map() 함수, sys 써보기 -런타임 에러
import sys
x, y = map(int, sys.stdin.readline().split())

if x>0:
    if y>0:
        print(1)
    if y<0:
        print(4)
if x<0:
    if y>0:
        print(2)
    if y<0:
        print(3)
        
# map함수, sys 쓰면 안되는 건지?????