
'''
--url--
https://www.acmicpc.net/problem/11654

--title--
11654번: 아스키 코드

--problem_description--
알파벳 소문자, 대문자, 숫자 0-9중 하나가 주어졌을 때, 주어진 글자의 아스키 코드값을 출력하는 프로그램을 작성하시오.

--problem_input--
알파벳 소문자, 대문자, 숫자 0-9 중 하나가 첫째 줄에 주어진다.

--problem_output--
입력으로 주어진 글자의 아스키 코드 값을 출력한다.
'''
print(chr(97))
a = input()

if type(a) == "<class 'str'>":
    a = ord(a)
    print(a)
if type(a) == "<class 'int'>":
    a = chr(a)
    print(a)
