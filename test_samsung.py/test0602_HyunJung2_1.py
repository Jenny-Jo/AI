# 1. 6/3 내일아침 삼성전자 시가 맞추기/ 시계열 예측 RNN
# 2. CSV 데이터는 건들지 말 것/날짜 거꾸로 된 것 바꾸기
# 3. 앙상블 모델 사용
# 4. 체크포인트/얼리스타핑/텐서보드/ 전부 다 응용하기
# 5. 6/2 오늘 저녁 6시까지 메일
# 하이트진로와 삼성전자 CSV 데이타
# 앙상블은 행은 같게, 열은 다르게

# 1. Data -CSV load하기/standard scaler/ pca /x, y 나누고/ train_test_split
import numpy as np
import pandas as pd
from keras.layers.core import Dropout

# 데이터 전처리 1//data 불러와서 dropna, hite는 bfill 까지
# 1
print('hite')
hite = pd.read_csv("./test_samsung.py/hite_stock.csv", index_col=0, header=0,sep=',')
print(hite)
print(hite.shape)       #(720,5)
print(hite.values)


hite = hite.dropna(axis=0, how='all')
hite = hite.fillna(method='bfill') 
print(hite.shape)        #(509,5)
print('hite: ', hite)


# 2
print('samsung')
samsung = pd.read_csv("./test_samsung.py/samsung_stock.csv", index_col=0, header=0, sep=',')
print(samsung)
print(samsung.shape)   #(700, 1)


samsung = samsung.dropna(axis = 0 )
print(samsung.shape)       #(509, 1)



# 데이터 전처리 2 // 문자형에서 실수화
for i in range(len(hite.index)):
    for j in range(len(hite.iloc[i])):
        hite.iloc[i,j] = int(hite.iloc[i,j].replace(',', ''))

for i in range(len(samsung.index)):
    for j in range(len(samsung.iloc[i])):
        samsung.iloc[i,j] = int(samsung.iloc[i,j].replace(',', ''))


# 데이터 전처리 3 // 일자 오름차순으로 정리
hite = hite.sort_values(['일자'], ascending=[True])
samsung = samsung.sort_values(['일자'], ascending=[True])
print(hite)
print(samsung)

# 데이터 값 저장
hite = hite.values
samsung = samsung.values

print(hite)
print(samsung)
print(type(hite)) # <class 'numpy.ndarray'> #(508, 5)
print(type(samsung)) # <class 'numpy.ndarray'> #(509, 1)

np.save('./test_samsung.py/hite.npy', arr = hite)
np.save('./test_samsung.py/samsung.npy', arr = samsung)

