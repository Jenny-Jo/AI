# DT & feature importance
# DT & feature importance

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.losses import MeanAbsoluteError
from sklearn.metrics import accuracy_score, r2_score
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor

# y_pred.to_csv(경로) >> submit 할 파일 만든다
# 회귀
# 인풋 71개 > y output 4개
# 평가는 mae / loss, metrics(mse or mae)에 포함

# 1. Data

# 1) 불러오기
train = pd.read_csv('./data/dacon/comp1/train.csv', header = 0, index_col = 0)
test = pd.read_csv('./data/dacon/comp1/test.csv', header = 0, index_col = 0)
submission = pd.read_csv('./data/dacon/comp1/sample_submission.csv', header = 0, index_col = 0)
# sep=',' 이게 디폴트

# 2) shape
print('train.shape : ', train.shape)
print('test.shape : ', test.shape)
print('submission.shape : ', submission.shape)

#  train.    shape :  (10000, 75) : 71 열은 x_train, x_test// 4 열은 y_train,y_test
#   test.    shape :  (10000, 71) : x_predict 
# submission.shape :  (10000, 4)  : y_predict 4

print(type(train))
import numpy as np
import pandas as pd  
import pandas as pd
from sklearn.feature_selection import SelectFromModel

def outliers_pd(data_out) :
    outliers = []
    for i in range(data_out.values.shape[1]):
        data = data_out.values[:, i]
        quartile_1, quartile_3 = np.percentile(i, [25,75])
        print(i,'1사분위 : ',quartile_1)
        print(i,'3사분위 : ',quartile_3)
        iqr = quartile_3 - quartile_1
        lower_bound = quartile_1 - (iqr * 1.5)
        upper_bound = quartile_3 + (iqr * 1.5)
        out = np.where((i > upper_bound) | (i<lower_bound))
        outliers.append(out)
    return outliers
    print('outliers',outliers)

a =outliers_pd(test)
print(a)



# 3) 결측치 보완
# print(train.isnull().sum())
# 3-1) interpolate
train = train.interpolate() #  보간법// 선형보간 -값들 전체의 선을 그리고 빈 자리를 선에 맞게 그려줌
test = test.interpolate() 

# 결측치 모두 처리 됨을 확인
# print(data.isnull().sum()) 
# print(x_pred.isnull().sum()) 


# 3-2) 첫행도 채워주기
train = train.fillna(method='bfill') 
test = test.fillna(method='bfill') 
# 3-3) 넘파이로 저장
# np.save(train, test)

# 3-4) 저장한거 로드하기

# 4) x, y 나눠주기
x1 = train.values[:, :71]
# x11 = train[:, :71]
x11 = train.loc[:,"rho":"990_dst"]
x_col = list(x11)
x11 = x11.values
# x_col = x11.columns
y1 = train.values[:, 71:]


x_predict = test.values
y_predict = submission.values


# 5) Standard scaler, PCA


scaler = StandardScaler()
scaler.fit(x1)
x1_scaled = scaler.transform(x1)
x_predict = scaler.transform(x_predict)



# from sklearn.decomposition import PCA
# pca = PCA(n_components=2)
# pca.fit(x1_scaled)
# x1 = pca.transform(x1_scaled)
# x_predict = pca.transform(x_predict)

y_predict = []

for i in range(4) :
    y_i = y1[:, i] 

    x_train, x_test, y_i_train, y_i_test = train_test_split(x1, y_i, train_size = 0.8)

    model = XGBRegressor()
    thresholds = np.sort(model.estimators_.feature_importances_)

    for thresh in thresholds : # thresh : feature importance 값 // 모델 선택함
    # AttributeError: 'GridSearchCV' object has no attribute 'estimators_'
        selection = SelectFromModel(model.estimators_[2], threshold=thresh, prefit=True)
                                                # median
        select_x_train = selection.transform(x_train)
        select_x_test = selection.transform(x_test)

        # print(select_x_train.shape)
        # GridSearch
        parameters = [
        { 'n_estimators': [90, 100, 110], 'learning_rate': [0.1, 0.001, 0.01],
        'max_depth':[4,5,6], 'colsample_bytree' : [0.6 ,0.9 ,1] }]

        # { 'n_estimators': [90, 110], 'learning_rate': [0.1, 0.001, 0.5],
        #     'max_depth':[4,5,6], 'colsample_bytree' : [0.6 ,0.9 ,1],
        #     'colsample_bylevel' : [0.6, 0.7, 0.9] }




    # 2. model

    # 3. fit
    model.fit(x_train, y_i_train)

    # 4. predict
    score = model.score(x_test, y_i_test)

    # print(model.feature_importances_) # 대박중요!!
    print("score:", score)
    y_predict.append(model.predict(x_predict))
print(y_predict)

# def plot_feature_importances_x11(model) :
#     n_features = x11.shape[1]
#     plt.barh(np.arange(n_features), model.feature_importances_,
#               align = 'center')
#     plt.yticks(np.arange(n_features), x_col)
#     plt.xlabel('Feature Importances')
#     plt.ylabel('Features')
#     plt.ylim(-1, n_features)
# plot_feature_importances_x11(model)
# plt.show()



# print(y_predict.shape)



y_pred = pd.DataFrame({
    'id' : np.array(range(10000, 20000)),
    'hhb': y_predict[0],
    'hbo2': y_predict[1],
    'ca': y_predict[2],
    'na': y_predict[3],
})
print(y_pred)

y_pred.to_csv('./dacon/comp1/y_pred.csv', index = False)
# model.save('./dacon/comp1/dacon1_model.save.h5')


# score: 0.6826087477365764
# score: 0.15263463846738812
# score: 0.14027518151046114
# score: 0.007097635717074446
