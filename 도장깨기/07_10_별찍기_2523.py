'''
https://www.acmicpc.net/problem/2523
문제
예제를 보고 규칙을 유추한 뒤에 별을 찍어 보세요.

입력
첫째 줄에 N(1 ≤ N ≤ 100)이 주어진다.

출력
첫째 줄부터 2×N-1번째 줄까지 차례대로 별을 출력한다.

예제 입력 1 
3
예제 출력 1 
*
**
***
**
*
'''

import sys
n = int(input())
number = int(2*n - 1)

for i in range(int(number/2)):
    print((i+1)*'*')
print('*'*int(number/2+1))
for i in (int(number/2),1):
    print((int(number)-1)*'*')    
    

