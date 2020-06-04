import numpy as np
import pandas as pd

# 1) header, index_col = 0 or 1?-------------------------------
samsung = pd.read_csv('./test_samsung.py/samsung_stock.csv',
                        index_col = 0, #None
                        header = 0,
                        sep=',',
                        )           

hite = pd.read_csv('./test_samsung.py/hite_stock.csv',
                    index_col = 0, #None
                    header = 0, # 판다스에서 값으로 인정안함
                    sep=',',
                    )

print(samsung.head())
print(hite.head())

# 2) 결측치 Nan 제거- slicing---------------------------------

'''
print(hite.head())
            시가      고가    저가    종가     거래량
일자
2020-06-02  39,000     NaN     NaN     NaN        NaN
2020-06-01  36,000  38,750  36,000  38,750  1,407,345
2020-05-29  35,900  36,750  35,900  36,000    576,566
2020-05-28  36,200  36,300  35,500  35,800    548,493
2020-05-27  35,900  36,450  35,800  36,400    373,464

print(samsung.tail())
      시가
일자
NaN  NaN
NaN  NaN
NaN  NaN
NaN  NaN
NaN  NaN
'''
# 시가 빼고 다 잘라버려도 괜찮음. 그러나, 큰 파일에선 안먹힘

# (1) dropna, fillna
print(hite.head())
print(samsung.tail())

samsung = samsung.dropna(axis=0)      # 행 삭제

hite = hite.fillna(method = 'bfill')  # 뒤에서 앞으로 채우기, 
                                      # colomn에 값을 주기 위해서 했다
hite = hite.dropna(axis = 0)

# (2) None 제거 2
hite = hite[0:509]
hite.iloc[0, 1:5] = [10,20,30,40]       # i : index, -1해줘야
# or
hite.loc["2020-06-02",'고가':'거래량'] = ['10', '20', '30', '40' ]
                                        # 정확히 거래량이라고 명시해야

'''
print(hite.head())
 2020-06-02  39,000      10      20      30         40
'''
# (3) Nan을 predict로 잡아도 됨 > 모르겠음

# 3) 삼성과 하이트의 정렬을 오름차순으로-------------------------------
samsung =samsung.sort_values(['일자'], ascending=['True'])
hite = hite.sort_values(['일자'], ascending=['True'])

print(samsung)
print(hite)

# 4) 콤마제거, 문자를 정수로 형 변환 ----------------------------------
for i in range(len(samsung.index)): #'37,000' str > 37000 int
    samsung.iloc[i, 0] = int(samsung.iloc[i, 0].replace(',', ''))
print(samsung)
print(samsung.shape)                          # (509, 1)
print(type(samsung.iloc[0,0])) # <class 'int'>

for i in range(len(hite.index)) :             # 행
    for j in range(len(hite.iloc[i])):        # 열
        hite.iloc[i, j] = int(hite.iloc[i, j].replace(',', ''))
print(hite)
print(hite.shape)                             # (509,5)

# 5) 판다스에서 넘파이로 바꾸기 ----------------------------------------
samsung = samsung.values
hite = hite.values

print(type(hite))                             # <class 'numpy.ndarray'>

np.save('./test_samsung.py/samsung.npy', arr=samsung)
np.save('./test_samsung.py/hite.npy', arr=hite)


