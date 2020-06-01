import numpy as np
import pandas as pd
from dask.array.routines import shape

datasets = pd.read_csv("./data/csv/iris.csv",
                        index_col=None, 
                        header=0, sep=',')
#index_col=None, 자동인덱스 해서 표기해줌 
#header=0, 헤더 취급 안해줌

print(datasets)
print(datasets.shape)
print(datasets.head())
print(datasets.tail())
print(datasets.values) # 판다스를 넘파이로 바꾸는게 .values 중요!!!

aaa= datasets.values
print(type(aaa))  #<class 'numpy.ndarray'>

# x, y슬라이싱하고 train_test_split(shuffle=True) 하고 저장


np.save('./data/csv.npy', arr=datasets)



'''
x = datasets[:, :4]
y = datasets[:, 4:]

print(x.shape)

     150    4  setosa  versicolor  virginica
0    5.1  3.5     1.4         0.2          0
1    4.9  3.0     1.4         0.2          0
2    4.7  3.2     1.3         0.2          0
3    4.6  3.1     1.5         0.2          0
4    5.0  3.6     1.4         0.2          0
..   ...  ...     ...         ...        ...
145  6.7  3.0     5.2         2.3          2
146  6.3  2.5     5.0         1.9          2
147  6.5  3.0     5.2         2.0          2
148  6.2  3.4     5.4         2.3          2
149  5.9  3.0     5.1         1.8          2
자동 인덱스
[150 rows x 5 columns]
'''
