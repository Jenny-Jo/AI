# gradient boost
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

# cmd > pip install xgboost
# 끝판왕이다
cancer = load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, train_size = 0.8, random_state=42
)

# model = DecisionTreeClassifier(max_depth=4) # 보통 3, 4 준다/ 5 주면 과적합/  layer 깊이와 비슷... decision tree가 node와 비슷
# model = RandomForestClassifier()
# model = GradientBoostingClassifier()
model = XGBClassifier()

# max_features :  기본값 써라
# n_estimators :  클수록 좋다. 단점은 메모리 너무 많이 차지, 기본값 100
# n_jobs = -1  :  병렬 처리


model.fit(x_train, y_train)

acc = model.score(x_test, y_test)

print(model.feature_importances_) # 대박중요!!
print(acc)
# [0.         0.         0.         0.         0.00639525 0.
#  0.         0.70458252 0.         0.         0.         0.        # 0.7 얘가 제일 영양가 있다
#  0.         0.01221069 0.         0.         0.0189077  0.0162341
#  0.         0.         0.05329492 0.05959094 0.05247428 0.
#  0.00940897 0.         0.         0.06690062 0.         0.        ]

import matplotlib.pyplot as plt
import numpy as np
'''
def plot_feature_importances_cancer(model):
    n_features = cancer.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_,
              align = 'center')
    plt.yticks(np.arange(n_features), cancer.feature_names)
    plt.xlabel('Feature Importances')
    plt.ylabel('Features')
    plt.ylim(-1, n_features)

plot_feature_importances_cancer(model)
plt.show()

'''

def plot_feature_importances_cancer(model) :
    n_features = cancer.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_,
              align = 'center')
    plt.yticks(np.arange(n_features), cancer.feature_names)
    plt.xlabel('Feature Importances')
    plt.ylabel('Features')
    plt.ylim(-1, n_features)
plot_feature_importances_cancer(model)
plt.show()

# dacon 에 적용하라... input, output만 알아두자


