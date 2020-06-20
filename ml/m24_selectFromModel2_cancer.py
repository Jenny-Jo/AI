

from sklearn.feature_selection import SelectFromModel
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, r2_score
# dataset = load_boston()
# x = dataset.data
# y = dataset.target

x, y = load_breast_cancer(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8,
                                                    shuffle = True, random_state = 66)

model = XGBClassifier()
model.fit(x_train, y_train)
score = model.score(x_test, y_test)
print('acc : ', score)

# feature engineering
thresholds = np.sort(model.feature_importances_)
print(thresholds)
# 오름차순으로 중요도가 낮은 애들부터 나옴

# 전체 컬럼수 13개만큼 돌림
for thresh in thresholds : # thresh : feature importance 값 // 모델 선택함
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
                                                # median
    select_x_train = selection.transform(x_train)
    
    print(select_x_train.shape)

# colomn이 하나씩, 중요하지 않은 애들부터 지움
# 중요한 애들 빼내

    selection_model =  XGBClassifier()
    selection_model.fit(select_x_train, y_train)      
    
    select_x_test = selection.transform(x_test)
    y_pred = selection_model.predict(select_x_test)
    
    score = r2_score(y_test, y_pred)
    print('acc : ', score)
    
    print('Thresh =%.3f, n=%d, R2 : %.2f%%' %(thresh,select_x_train.shape[1],
                                              score*100))

'''
acc :  0.9736842105263158
[0.         0.         0.00037145 0.00233393 0.00278498 0.00281184 
 0.00326043 0.00340272 0.00369179 0.00430626 0.0050556  0.00513449 
 0.0054994  0.0058475  0.00639412 0.00769184 0.00775311 0.00903706 
 0.01171023 0.0136856  0.01420499 0.01813928 0.02285903 0.02365488 
 0.03333857 0.06629944 0.09745205 0.11586285 0.22248562 0.28493083]
(455, 30)
acc :  0.885733377881724
Thresh =0.000, n=30, R2 : 88.57%
(455, 30)
acc :  0.885733377881724
Thresh =0.000, n=30, R2 : 88.57%
(455, 28)
acc :  0.885733377881724
Thresh =0.000, n=28, R2 : 88.57%
(455, 27)
acc :  0.885733377881724
Thresh =0.002, n=27, R2 : 88.57%
(455, 26)
acc :  0.885733377881724
Thresh =0.003, n=26, R2 : 88.57%
(455, 25)
acc :  0.885733377881724
Thresh =0.003, n=25, R2 : 88.57%
(455, 24)
acc :  0.885733377881724
Thresh =0.003, n=24, R2 : 88.57%
(455, 23)
acc :  0.885733377881724
Thresh =0.003, n=23, R2 : 88.57%
(455, 22)
acc :  0.8476445038422986
Thresh =0.004, n=22, R2 : 84.76%
(455, 21)
acc :  0.8476445038422986
Thresh =0.004, n=21, R2 : 84.76%
(455, 20)
acc :  0.885733377881724
Thresh =0.005, n=20, R2 : 88.57%
(455, 19)
acc :  0.885733377881724
Thresh =0.005, n=19, R2 : 88.57%
(455, 18)
acc :  0.8476445038422986
Thresh =0.005, n=18, R2 : 84.76%
(455, 17)
acc :  0.8476445038422986
Thresh =0.006, n=17, R2 : 84.76%
(455, 16)
acc :  0.8476445038422986
Thresh =0.006, n=16, R2 : 84.76%
(455, 15)
acc :  0.885733377881724
Thresh =0.008, n=15, R2 : 88.57%
(455, 14)
acc :  0.885733377881724
Thresh =0.008, n=14, R2 : 88.57%
(455, 13)
acc :  0.9238222519211493
Thresh =0.009, n=13, R2 : 92.38%
(455, 12)
acc :  0.9238222519211493
Thresh =0.012, n=12, R2 : 92.38%
(455, 11)
acc :  0.9238222519211493
Thresh =0.014, n=11, R2 : 92.38%
(455, 10)
acc :  0.9238222519211493
Thresh =0.014, n=10, R2 : 92.38%
(455, 9)
acc :  0.885733377881724
Thresh =0.018, n=9, R2 : 88.57%
(455, 8)
acc :  0.885733377881724
Thresh =0.023, n=8, R2 : 88.57%
(455, 7)
acc :  0.9238222519211493
Thresh =0.024, n=7, R2 : 92.38%
(455, 6)
acc :  0.885733377881724
Thresh =0.033, n=6, R2 : 88.57%
(455, 5)
acc :  0.8095556298028733
Thresh =0.066, n=5, R2 : 80.96%
(455, 4)
acc :  0.8476445038422986
Thresh =0.097, n=4, R2 : 84.76%
(455, 3)
acc :  0.7714667557634479
Thresh =0.116, n=3, R2 : 77.15%
(455, 2)
acc :  0.6191112596057466
Thresh =0.222, n=2, R2 : 61.91%
(455, 1)
acc :  0.5048446374874705
Thresh =0.285, n=1, R2 : 50.48%
'''