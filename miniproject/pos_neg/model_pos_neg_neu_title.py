# 파일에서 단어를 불러와 posneg리스트를 만드는 코드

출처: https://somjang.tistory.com/entry/Keras기사-제목을-가지고-긍정-부정-중립-분류하는-모델-만들어보기 [솜씨좋은장씨]
import codecs
positive = [] 
negative = [] 
posneg = [] 
pos = codecs.open("./positive_words_self.txt", 'rb', encoding='UTF-8') 

while True: 
    line = pos.readline() 
    line = line.replace('\n', '') 
    positive.append(line) 
    posneg.append(line) 

    if not line: break 
pos.close() 

neg = codecs.open("./negative_words_self.txt", 'rb', encoding='UTF-8') 

while True: 
    line = neg.readline() 
    line = line.replace('\n', '') 
    negative.append(line) 
    posneg.append(line) 

    if not line: break

neg.close()

# 출처: https://somjang.tistory.com/entry/Keras기사-제목을-가지고-긍정-부정-중립-분류하는-모델-만들어보기 [솜씨좋은장씨]

# 크롤링한 기사 제목과 기사 제목과 posneg를 활용하여 만든 긍정(1), 부정(-1), 중립(0)라벨 정보를 가지는 dataframe을 만드는 함수
# (예시 : 네이버에서 버거킹으로 검색하여 나온 기사 4,000개 제목과 각각 제목의 긍정, 부정, 중립 라벨 생성)


import requests 
from bs4 import BeautifulSoup 
import re import pandas as pd 

label = [0] * 4000 
my_title_dic = {"title":[], "label":label} 
j = 0

for i in range(400): 
    num = i * 10 + 1 
    # bhc
    #  # url = "https://search.naver.com/search.naver?&where=news&query=bhc&sm=tab_pge&sort=0&photo=0&field=0&reporter_article=&pd=0&ds=&de=&docid=&nso=so:r,p:all,a:all&mynews=0&cluster_rank=23&start=" + str(num) 
    # # 아오리라멘 # url2 = "https://search.naver.com/search.naver?&where=news&query=%EC%95%84%EC%98%A4%EB%A6%AC%EB%9D%BC%EB%A9%98&sm=tab_pge&sort=0&photo=0&field=0&reporter_article=&pd=0&ds=&de=&docid=&nso=so:r,p:all,a:all&mynews=0&cluster_rank=34&start=" + str(num) 
    # # 버거킹 
    url3 = "https://search.naver.com/search.naver?&where=news&query=%EB%B2%84%EA%B1%B0%ED%82%B9&sm=tab_pge&sort=0&photo=0&field=0&reporter_article=&pd=0&ds=&de=&docid=&nso=so:r,p:all,a:all&mynews=0&cluster_rank=23&start=" + str(num) 
    req = requests.get(url3) 
    soup = BeautifulSoup(req.text, 'lxml') 
    titles = soup.select("a._sp_each_title")

for title in titles:
    title_data = title.text
    title_data = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…\"\“》]', '', title_data) 
    my_title_dic['title'].append(title_data)

출처: https://somjang.tistory.com/entry/Keras기사-제목을-가지고-긍정-부정-중립-분류하는-모델-만들어보기 [솜씨좋은장씨]