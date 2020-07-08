# 그룹별로 나누고, 집계함수 적용, 집계 결과 하나로 합침
import pandas as pd
df = pd.DataFrame({'상품번호' : ['P1', 'P1', 'P2', 'P2'],
                   '수량' :     [2, 3, 5, 10]})
'''
# min 상품번호별 판매된 최소 수량
df = df.groupby(by='상품번호', as_index=False).min()
print(df)
#   상품번호  수량
# 0   P1   2
# 1   P2   5'''

# max 상품번호별 판매된 최대 수량
df2 = df.groupby(by=['상품번호'], as_index=False).max()
print(df2)

#   상품번호  수량
# 0   P1   3
# 1   P2  10

# count : 상품번호별 판매 수
df3 = df.groupby(by=['상품번호'], as_index=False).count()
print(df3)
#   상품번호  수량
# 0   P1   2
# 1   P2   2

# sum 상품번호별 총 수량
df4 = df.groupby(by=['상품번호'], as_index=False).sum()
print(df4)
#   상품번호  수량
# 0   P1   5
# 1   P2  15

# mean : 상품번호별 평균 수량
df5 = df.groupby(by=['상품번호'], as_index=False).mean()
print(df5)
#   상품번호   수량
# 0   P1  2.5
# 1   P2  7.5

## 여러개 열을 기준으로 집계하기
df11 = pd.DataFrame({'고객번호' : ['C1', 'C2', 'C2', 'C2'],
                   '상품번호' : ['P1', 'P1', 'P2', 'P2'],
                   '수량' :     [2, 3, 5, 10]})

# sum : 고객별 발송해야 할 상품별 수량 합계
df12 = df11.groupby(by=['고객번호', '상품번호'], as_index=False).sum()
print(df12)
#   고객번호 상품번호  수량
# 0   C1   P1   2
# 1   C2   P1   3
# 2   C2   P2  15


