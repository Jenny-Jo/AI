import numpy as np

import pandas as pd
 

# 판다스로 불러들인다.
datasets = pd.read_csv("./data/csv/data_sam.csv", index_col = None,
                        header = 0, encoding='cp949', sep =',')

datasets = pd.read_csv("./data/csv/data_hit.csv", index_col = None,
                        header = 0, encoding='cp949', sep =',')
print(datasets)

print("====판다스를 넘파이로 바꾸는 함수(.values)========")
print(datasets.values)

aaa = datasets.values
bbb = datasets.values

print(type(aaa))  # <class 'numpy.ndarray'>
print(type(bbb))  # <class 'numpy.ndarray'>
