#백준 문제 긁어오기

from urllib.request import urlopen
import bs4
import pandas as pd
from random import randint

# index = "https://finance.naver.com/sise/sise_index_day.nhn?code="+index_cd+"&page="+str(page_n)
q_num=randint(1,10001)
index = f"https://www.acmicpc.net/problem/{q_num}"
source = urlopen(index).read()
source_bs4 = bs4.BeautifulSoup(source,"html.parser")

# print(source)
title = source_bs4.find('title').string
print()
print("--title--")
print(f"{title}")

print()
print("--problem_description--")
text_0 = source_bs4.find('div', id ="problem_description").find_all('p')
for t in text_0:
    print(t.string)
    print()

print()
print("--problem_input--")
text_0 = source_bs4.find('div', id ="problem_input").find_all('p')
for t in text_0:
    print(t.string)
    print()

print()
print("--problem_output--")
text_0 = source_bs4.find('div', id ="problem_output").find_all('p')
for t in text_0:
    print(t.string)
    print()

