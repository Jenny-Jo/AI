

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

model = XGBClassifier(n_estimators = 100, learning_rate = 0.05, n_jobs = -1)
model.fit(x_train, y_train)

# feature engineering
thresholds = np.sort(model.feature_importances_)
print(thresholds)
# 오름차순으로 중요도가 낮은 애들부터 나옴

# 전체 컬럼수 13개만큼 돌림
for thresh in thresholds : # thresh : feature importance 값 // 모델 선택함
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
      
    select_x_test = selection.transform(x_test)                                           # median
    select_x_train = selection.transform(x_train)
    
    print(select_x_train.shape)

# colomn이 하나씩, 중요하지 않은 애들부터 지움
# 중요한 애들 빼내

    selection_model =  XGBClassifier(n_estimators = 100, learning_rate = 0.05, n_jobs = -1)
    selection_model.fit(x_train, y_train, verbose = True, eval_metric= ['logloss', 'auc'],
                            eval_set= [(x_train, y_train),(x_test, y_test)],
                            early_stopping_rounds= 20) # fit을 안에 넣은 이유?
    select_x_test = selection.transform(x_test)
    y_pred = selection_model.predict(select_x_test)
    
    score = r2_score(y_test, y_pred)
    print('acc : ', score)
    
    print('Thresh =%.3f, n=%d, R2 : %.2f%%' %(thresh,select_x_train.shape[1],
                                              score*100))

