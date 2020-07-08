'''
--url--
https://www.acmicpc.net/problem/2747

--title--
2747번: 피보나치 수

--problem_description--
피보나치 수는 0과 1로 시작한다. 0번째 피보나치 수는 0이고, 1번째 피보나치 수는 1이다. 그 다음 2번째 부터는 바로 앞 두 피보나치 수의 합이 된다.

None

n=17일때 까지 피보나치 수를 써보면 다음과 같다.

0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597

n이 주어졌을 때, n번째 피보나치 수를 구하는 프로그램을 작성하시오.

--problem_input--
첫째 줄에 n이 주어진다. n은 45보다 작거나 같은 자연수이다.

--problem_output--
첫째 줄에 n번째 피보나치 수를 출력한다.

'''

# N차 시도 - 런타임 에러
import sys
n = int(sys.stdin.readline())

Fibonacci = []
if n>=1 and n<=46:
    for i in range(n):
        if n == 1 :
            Fibonacci.append(0)
        if n == 2 :
            Fibonacci.wappend(1)
        elif n >=3  and n <= 46 :
            num = Fibonacci[n-2]+Fibonacci[n-1]
            Fibonacci.append(num)
print(Fibonacci[n-1])
        
        
        
        
    
    