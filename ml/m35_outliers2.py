'''
import numpy as np
import pandas as pd

def outliers(data_out) :
    quartile_1, quartile_3 = np.percentile(data_out, [25,75])
    print('1사분위 : ',quartile_1)
    print('3사분위 : ',quartile_3)
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return np.where((data_out > upper_bound) | (data_out<lower_bound))

a = np.array([1,2,3,4,10000,6,7,5000,90,100])
# (10, )

b = outliers(a)
print('이상치의 위치 : ',b)

# 실습 : 행렬 입력하여 컬럼별로 이상치 발견하는 함수 구하기
# 컬럼별로 나올 수 있게 함수 만들기...?

'''

# 넘파이
import numpy as np
import pandas as pd  

def outliers_np(data_out) :
    outliers = []
    for i in range(data_out.shape[1]):
        data = data_out[:, i]
        quartile_1, quartile_3 = np.percentile(data, [25,75])
        iqr = quartile_3 - quartile_1
        print(i,'1사분위 : ',quartile_1)
        print(i,'3사분위 : ',quartile_3)
        lower_bound = quartile_1 - (iqr * 1.5)
        upper_bound = quartile_3 + (iqr * 1.5)
        out = np.where((data > upper_bound) | (data<lower_bound))
        outliers.append(out)
    print('outliers :' , outliers)
    return outliers
    

a = np.array([[10,20,30,50,100],
             [10,300,20,50,60],
             [20,30,40,50,60],
             [50,60,80,900,1],
             [100,90,100,110,900]])

print(a)

b = outliers_np(a)
print('b : ',b)


# 판다스
def outliers_pd(data_out):
        quartile_1 = data_out.quantile(.25)
        quartile_3 = data_out.quantile(.75)
        print("1사 분위 : ",quartile_1)                                       
        print("3사 분위 : ",quartile_3)                                        
        iqr = quartile_3 - quartile_1
        lower_bound = quartile_1 - (iqr * 1.5)
        upper_bound = quartile_3 + (iqr * 1.5)
        return np.where((data_out > upper_bound) | (data_out < lower_bound))
         
    
c = {'fruits' : ['a','b','c'], 'time' : [1, 2, 3], 'year' : [100,200,1000]}


df = pd.DataFrame(c)
print(df.values)
d = outliers_pd(df)
print('d: ',d)
    
    


