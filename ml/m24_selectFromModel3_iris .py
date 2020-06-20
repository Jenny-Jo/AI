

from sklearn.feature_selection import SelectFromModel
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, r2_score
# dataset = load_boston()
# x = dataset.data
# y = dataset.target

x, y = load_iris(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8,
                                                    shuffle = True, random_state = 66)

model = XGBRegressor()
model.fit(x_train, y_train)
score = model.score(x_test, y_test)
print('acc : ', score)

# feature engineering
thresholds = np.sort(model.feature_importances_)
print(thresholds)
# 오름차순으로 중요도가 낮은 애들부터 나옴

# R2 :  0.9221188544655419
# [0.00134153 0.00363372 0.01203115 0.01220458 0.01447935 0.01479119
#  0.0175432  0.03041655 0.04246345 0.0518254  0.06949984 0.30128643
#  0.42848358]

# 전체 컬럼수 13개만큼 돌림
for thresh in thresholds : # thresh : feature importance 값 // 모델 선택함
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
                                                # median
    select_x_train = selection.transform(x_train)
    
    # print(select_x_train.shape)
    '''
(404, 13)
(404, 12)
(404, 11)
(404, 10)
(404, 9)
(404, 8)
(404, 7)
(404, 6)
(404, 5)
(404, 4)
(404, 3)
(404, 2)
(404, 1)
'''
# colomn이 하나씩, 중요하지 않은 애들부터 지움
# 중요한 애들 빼내

    selection_model =  XGBRegressor()
    selection_model.fit(select_x_train, y_train)      
    
    select_x_test = selection.transform(x_test)
    y_pred = selection_model.predict(select_x_test)
    
    score = r2_score(y_test, y_pred)
    print('acc : ', score)
    
    print('Thresh =%.3f, n=%d, R2 : %.2f%%' %(thresh,select_x_train.shape[1],
                                              score*100))

'''
acc :  0.8608475814759798
[0.00744195 0.01018879 0.15876685 0.82360244]
acc :  0.8608475814759797
Thresh =0.007, n=4, R2 : 86.08%
acc :  0.8385032389347137
Thresh =0.010, n=3, R2 : 83.85%
acc :  0.9341640188302681
Thresh =0.159, n=2, R2 : 93.42%
acc :  0.9430670036443007
Thresh =0.824, n=1, R2 : 94.31%
'''