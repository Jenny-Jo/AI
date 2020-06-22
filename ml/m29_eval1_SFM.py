from sklearn.feature_selection import SelectFromModel
import numpy as np
from xgboost import XGBClassifier, XGBRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.metrics import accuracy_score, r2_score

x, y = load_boston(return_X_y=True)
print(x.shape)      # (506, 13)
print(y.shape)      # (506, )

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8,
                 shuffle = True, random_state = 66)

model = XGBRegressor(n_estimators=300, learning_rate=0.1, n_jobs = -1)
                    # = epochs
                                                     # loss
model.fit(x_train, y_train)

thresholds = np.sort(model.feature_importances_)
print(thresholds)

for thresh in thresholds : # thresh : feature importance 값 // 모델 선택함
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
                                                # median
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)
    # print(select_x_train.shape)


    selection_model =  XGBRegressor(n_estimators = 100, learning_rate = 0.05, n_jobs = -1)
    selection_model.fit(x_train, y_train, verbose=True, eval_metric=['logloss','rmse'],
                eval_set=[(x_train, y_train), (x_test, y_test)], # rmse, mae, logloss, error, auc
                early_stopping_rounds= 100)      ## 여기에 들어가는 이유??
    
    y_pred = selection_model.predict(x_test)
    
    score = r2_score(y_test, y_pred)
    print('R2 : ', score)
    
    print('Thresh =%.3f, n=%d, R2 : %.2f%%' %(thresh,select_x_train.shape[1],
                                              score*100))
