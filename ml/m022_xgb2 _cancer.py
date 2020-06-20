

# 과적합 방지
# 1. 훈련데이터량을 늘린다.
# 2. 피처수를 줄인다.
# 3. regularization
# 점수 :  0.9736842105263158

from xgboost import XGBClassifier, plot_importance
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC


dataset = load_breast_cancer()
x = dataset.data
y = dataset.target

print(x.shape) # (506, 13)
print(y.shape) # (506,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8,
                                    shuffle = True, random_state = 66 )


# XGB 필수 파라미터 ##
n_estimators = 10000        #나무 100개/ decision tree보다 100배 느려
learning_rate = 0.001      # 학습률 디폴트 값 0.01 / 상당히 중요하다
colsample_bytree = 0.9
colsample_bylevel = 0.6   # 0.6~0.9
###
# 점수 :  0.9736842105263158

max_depth = 5
n_jobs = -1 #  딥러닝 빼고 default 로 써라

# 트리- 전처리 안해, 결측치 제거 안해, 
# xgb - 속도 빨라, 일반 머신러닝보단 조금 느려/ 앙상블이라서
# 보간법 안해도 돼

# CV, feature importance 꼭 넣기

model = XGBClassifier(max_depth= max_depth, learning_rate=learning_rate, n_estimators=n_estimators, n_jobs=n_jobs, 
                       #colsample_bylevel = colsample_bylevel, 
                       colsample_bytree= colsample_bytree)

model.fit(x_train, y_train)

score = model.score(x_test, y_test) # evaluate
print('점수 : ', score)
print(model.feature_importances_)

plot_importance(model)
# plt.show()

# f0~ f12 까지,  f12가 제일 중요

# 점수 :  -0.06904014604139475
# [0.03427136 0.00086752 0.01226326 0.         0.06205949 0.35779986
#  0.00939034 0.05117243 0.00469103 0.01540447 0.06899065 0.01284792
#  0.37024173]

