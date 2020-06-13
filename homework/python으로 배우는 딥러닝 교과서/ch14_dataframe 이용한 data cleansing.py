# 1.1 pandas로 csv 읽기
import pandas as pd

df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header= None)
df.columns = [ "sepal length", "sepal width","petal length", "petal width", "class"]
print(df)

'''
     sepal length  sepal width  petal length  petal width           class
0             5.1          3.5           1.4          0.2     Iris-setosa
1             4.9          3.0           1.4          0.2     Iris-setosa
2             4.7          3.2           1.3          0.2     Iris-setosa
3             4.6          3.1           1.5          0.2     Iris-setosa
4             5.0          3.6           1.4          0.2     Iris-setosa
..            ...          ...           ...          ...             ...
145           6.7          3.0           5.2          2.3  Iris-virginica
146           6.3          2.5           5.0          1.9  Iris-virginica
147           6.5          3.0           5.2          2.0  Iris-virginica
148           6.2          3.4           5.4          2.3  Iris-virginica
149           5.9          3.0           5.1          1.8  Iris-virginica

[150 rows x 5 columns]
'''
# 1.2 csv library로 csv 만들기
import csv

with open("olympic.csv","w") as csvfile:
    writer = csv.writer(csvfile, lineterminator="\n")

    writer.writerow(['city', 'year', 'season'])
    writer.writerow(['beijing', '2008', 'summer'])
    writer.writerow(['peongchang','2018', 'winter'])
    writer.writerow(['seoul', '1988', 'summer'])

print(pd.read_csv('olympic.csv'))
#          city  year  season
# 0     beijing  2008  summer
# 1  peongchang  2018  winter
# 2       seoul  1988  summer

# 1.3 pandas로 csv 만들기  > 더 만들기 쉬움
import pandas as pd

data = { 'Name' : ['Galaxy', 'iphone'], 
        'realease' : [2009, 2007 ],
        'country' : ['Korea', 'USA']}
df = pd.DataFrame(data)
df.to_csv('smartphone.csv')
print(pd.read_csv('smartphone.csv'))

# 2. data frame 복습
import pandas as pd
from pandas import Series , DataFrame

birth = {'ID':['100','101','102','103'],
         'city':['Seoul','Busan','Bucheon', 'ChunCheon'],
         'year' :['1990','1998','1992','1993'],
         'name' : ['현정','태식','정우','아영']
}
birth_df = DataFrame (birth)

birth2 ={ 'ID' : ['104','105'],
        'city' : ['NYC', 'Paris'],
        'year' : ['1990','1999'],
        'name' : ['Christina','Ombli']

}
birth2_df = DataFrame(birth2)

append_df = birth_df.append(birth2_df).sort_values(by='ID', ascending= True)#.reset_index(drop=True)
print(append_df)

# 3. missing value
# 3.1 삭제

# listwise 삭제 : .dropna() 결측치 있는 행이 삭제
# pairwise deletion : 결손 적은 열 남기기
import numpy as np
from numpy import nan as NA
import pandas as pd

np.random.seed(0)

sample_df = pd.DataFrame(np.random.rand(10,4))

sample_df.iloc[1,0] = NA
sample_df.iloc[2,2] = NA
sample_df.iloc[5:,3] = NA

# pairwise deletion
print(sample_df[[0,2]].dropna())

# 3.2 fillna 앞/뒤에서 끌어다가 채우기
print(sample_df.fillna(method='ffill'))

# 3.3 평균값 대입
sample_df.fillna(sample_df.mean())

sample2_df = pd.DataFrame(np.random.rand(10,4))
sample2_df.iloc[1,0]=NA
sample2_df.iloc[6:,2]=NA
print(sample2_df)
print(sample2_df.fillna(sample2_df.mean()))

# 4. 데이터요약

# 4.1 key별 통계량 산출
import pandas as pd
df = pd.read_csv("F:\Study\ml\winequality-white.csv", header=None, sep=';')
df = DataFrame(df)
print(df)
df.columns = [ "fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol", "quality"
]
print(df)
# print('fixed acidity mean : ', df['fixed acidity'].mean())
# 에러 모르겠음...

# 4.2 중복 데이터 삭제 .drop_duplicates())
import pandas as pd
from pandas import DataFrame

dupli_data = DataFrame({'col1' : [1,1,2,3,4,4,6,6,7,7,7,8,9,9],
                         'col2' : ['a','b','b','b','c','c','b','b','d','d','c','b','c','c']})
print(dupli_data)
print(dupli_data.drop_duplicates())

# 4.3 mapping // .map(새로운 변수)


birth = {'ID':['100','101','102','103'],
         'city':['Seoul','Busan','Bucheon', 'ChunCheon'],
         'year' :[1990,1998,1986,2001],
         'name' : ['현정','태식','정우','아영'] }

birth = DataFrame(birth) # 이거 빠뜨리면 리스트형만 뜸
print(birth)
city_map = {'Seoul':'Seoul',
           'Busan' : 'Busan',
           'Bucheon' : 'KyeonKi',
           'ChunCheon' : 'GangWon'}

birth['region'] = birth['city'].map(city_map)
print(birth)

# # 4.4 구간 분할 
year_bins = [1985,1990,1995,2000,2005]
cut_data = pd.cut(birth.year, year_bins)                   # int
# fa_cut = pd.cut(df_ten["fixed acidity"], f_acidity)      # str

print(cut_data)
# ()포함 X. [] 포함 O
# 0    (1985, 1990]
# 1    (1995, 2000]
# 2    (1985, 1990]
# 3    (2000, 2005]
# Name: year, dtype: category
# Categories (4, interval[int64]): [(1985, 1990] < (1990, 1995] < (1995, 2000] < (2000, 2005]]

# 구간 갯수세기
print(pd.value_counts(cut_data))
# (1985, 1990]    2
# (2000, 2005]    1
# (1995, 2000]    1
# (1990, 1995]    0
# Name: year, dtype: int64

# 이름붙이기
group_names = ['second 80', 'first 90','second 90','first 2000']
cut_data = pd.cut (birth.year, year_bins, labels = group_names)
print(pd.value_counts(cut_data))

# second 80     2
# first 2000    1
# second 90     1
# first 90      0
# Name: year, dtype: int64

# 분할수 지정
print(pd.cut (birth.year, 10))
birth = {'ID':[100,101,102,103],
         'city':['Seoul','Busan','Bucheon', 'ChunCheon'],
         'year' :[1990,1998,1986,2001],
         'name' : ['현정','태식','정우','아영'] }
birth = DataFrame(birth) 

print(pd.cut(birth.year, 2))

# =====================================================
# 연습문제
df = pd.read_csv('.\ml\winequality-white.csv', header=0, sep=';')
df_ten = df.head(10)

df_ten.iloc[1,0] = NA
df_ten.iloc[2,3] = NA
df_ten.iloc[4,8] = NA
df_ten.iloc[7,3] = NA

df_ten = df_ten.fillna(df_ten.mean())

print('mean : ', df_ten['fixed acidity'].mean())

print('append: ',df_ten.append(df_ten.loc[3]))
df_ten.append(df_ten.loc[6])
df_ten.append(df_ten.loc[9])
df_ten = df_ten.drop_duplicates()
print(df_ten)

f_acidity= [5,6,7,8,9]
fa_cut = pd.cut(df_ten["fixed acidity"], f_acidity)
print(fa_cut)

print(pd.value_counts(fa_cut))
# (6, 7]    4
# (8, 9]    3
# (7, 8]    3
# (5, 6]    0
# Name: fixed acidity, dtype: int64
