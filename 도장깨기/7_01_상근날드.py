'''
--url--
https://www.acmicpc.net/problem/5543

--title--
5543번: 상근날드

--problem_description--
상근날드에서 가장 잘 팔리는 메뉴는 세트 메뉴이다. 주문할 때, 자신이 원하는 햄버거와 음료를 하나씩 골라,
 세트로 구매하면, 가격의 합계에서 50원을 뺀 가격이 세트 메뉴의 가격이 된다.

None

햄버거와 음료의 가격이 주어졌을 때, 가장 싼 세트 메뉴의 가격을 출력하는 프로그램을 작성하시오.

--problem_input--
입력은 총 다섯 줄이다. 첫째 줄에는 상덕버거, 둘째 줄에는 중덕버거, 셋째 줄에는 하덕버거의 가격이 주어진다. 넷째 줄에는 콜라의 가격, 다섯째 줄에는 사이다의 가격이 주어진다. 
모든 가격은 100원 이상, 2000원 이하이다.

--problem_output--
첫째 줄에 가장 싼 세트 메뉴의 가격을 출력한다.
'''
# N차 시도 - 맞았습니다!! 문제의 조건을 자꾸 빼먹어서 틀림. 다음부턴 조건부터 정리하고 문제 풀어야 겠음

# 세트메뉴
# 햄버거와 음료 하나씩 골라
# 가격 합계에서 -50
# 가장 싼

import sys
burgers =[]
for burger in range(3):
    burger = int(sys.stdin.readline())
    burgers.append(burger)
burgers.sort()

soft_drinks = []
for soft_drink in range(2):
    soft_drink = int(sys.stdin.readline())
    soft_drinks.append(soft_drink)
soft_drinks.sort()

cheapest_set_menu = burgers[0] + soft_drinks[0] -50
cheapest_set_menu = min(burgers) + min(soft_drinks) -50

print(cheapest_set_menu)

# min 함수 써서 해도 됨



