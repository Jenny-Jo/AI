'''
import numpy as np
import pandas as pd

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


hite = hite[0:509]
hite.iloc[0, 1:5] = [10,20,30,40]       # i : index, -1해줘야
# or
hite.loc["2020-06-02",'고가':'거래량'] = ['10', '20', '30', '40' ]
                                        # 정확히 거래량이라고 명시해야


import numpy as np
import pandas as pd

samsung = pd.read_csv('./test_samsung.py/samsung_stock.csv', index_col=0, header=0, sep=',')
hite = pd.read_csv('./test_samsung.py/hite_stock.csv', index_col = 0, header = 0, sep=',')

hite = hite[0:509]
# hite.iloc[0, 1:] = [1000,2000,3000,4000]
hite.loc["2020-06-02", '고가':'거래량'] = ['1000','2000', '3000', '4000']
print(hite)

# (1) dropna, fillna
print(hite.head())
print(samsung.tail())

samsung = samsung.dropna(axis=0)      # 행 삭제

hite = hite.fillna(method = 'bfill')  # 뒤에서 앞으로 채우기, 
                                      # colomn에 값을 주기 위해서 했다
hite = hite.dropna(axis = 0)

'''

import numpy as np
import pandas as pd

samsung = pd.read_csv('./test_samsung.py/samsung_stock.csv', index_col=0, header=0, sep=',')
hite = pd.read_csv('./test_samsung.py/hite_stock.csv', index_col = 0, header = 0, sep=',')

samsung = samsung.dropna(axis=0)
hite = hite.fillna(method='bfill')
hite = hite.dropna(axis = 0)
print(hite)
print(samsung)