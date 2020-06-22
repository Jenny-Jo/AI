

from sklearn.feature_selection import SelectFromModel
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.datasets import load_boston
from sklearn.metrics import accuracy_score, r2_score
# dataset = load_boston()
# x = dataset.data
# y = dataset.target

x, y = load_boston(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8,
                                                    shuffle = True, random_state = 66)

parameters = [
    { 'n_estimators': [100, 200, 300], 'learning_rate': [0.1, 0.3, 0.5, 0.01],
     'max_depth':[4,5,6] },
    
    { 'n_estimators': [90, 100, 110], 'learning_rate': [0.1, 0.001, 0.01],
     'max_depth':[4,5,6], 'colsample_bytree' : [0.6 ,0.9 ,1] },
    
    { 'n_estimators': [90, 110], 'learning_rate': [0.1, 0.001, 0.5],
     'max_depth':[4,5,6], 'colsample_bytree' : [0.6 ,0.9 ,1],
     'colsample_bylevel' : [0.6, 0.7, 0.9] }

]
n_jobs = -1

model = GridSearchCV(XGBRegressor(), parameters, cv=5, n_jobs=-1) # n_jobs 는 속도 높이기 위해 그리드 서치에 통으로 돌린다

model.fit(x_train, y_train)
print('-------------------------------------')
print(model.best_estimator_)
print('-------------------------------------')
print(model.best_params_)
print('-------------------------------------')


score = model.score(x_test, y_test)
print('R2 : ', score)


#### feature engineering ####

thresholds = np.sort(model.feature_importances_)
print(thresholds)

for thresh in thresholds : # thresh : feature importance 값 // 모델 선택함
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
                                                # median
    select_x_train = selection.transform(x_train)
    

    selection_model =  XGBRegressor()
    selection_model.fit(select_x_train, y_train)      
    
    select_x_test = selection.transform(x_test)
    y_pred = selection_model.predict(select_x_test)
    
    score = r2_score(y_test, y_pred)
    print('R2 : ', score)
    
    print('Thresh =%.3f, n=%d, R2 : %.2f%%' %(thresh,select_x_train.shape[1],
                                              score*100))
