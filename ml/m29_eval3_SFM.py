from xgboost import XGBClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectFromModel
import numpy as np

x, y = load_iris(return_X_y=True)

print(x.shape)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8,
                                                    shuffle = True, random_state = 66)

model = XGBClassifier(objective='multi:softmax', n_estimators = 100, learning_rate = 0.05, n_jobs = -1)

model.fit(x_train, y_train, verbose = True, eval_metric= ['mlogloss', 'merror'],
                            eval_set= [(x_train, y_train),(x_test, y_test)],
                            early_stopping_rounds= 20)

result = model.evals_result()
print("eval's result : ", result)

y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred.round())
print('Accuracy : ', acc)

epochs = len(result['validation_0']['mlogloss'])
x_axis = range(0, epochs)

fig, ax = plt.subplots()
ax.plot(x_axis, result['validation_0']['mlogloss'], label = 'Train')
ax.plot(x_axis, result['validation_1']['mlogloss'], label = 'Test')
ax.legend()
plt.ylabel('Log Loss')
plt.title('XGBbosst Log Loss')

ax.plot(x_axis, result['validation_0']['merror'], label = 'Train')
ax.plot(x_axis, result['validation_1']['merror'], label = 'Test')
ax.legend()
plt.ylabel('error')
plt.title('XGBbosst error')

# feature engineering
thresholds = np.sort(model.feature_importances_)
print(thresholds)
for thresh in thresholds : # thresh : feature importance 값 // 모델 선택함
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
                                                # median
    select_x_train = selection.transform(x_train)
    
    # print(select_x_train.shape)
    '''

'''
# colomn이 하나씩, 중요하지 않은 애들부터 지움
# 중요한 애들 빼내

    selection_model =  XGBClassifier()
    selection_model.fit(select_x_train, y_train)      
    
    select_x_test = selection.transform(x_test)
    y_pred = selection_model.predict(select_x_test)
    
    score = accuracy_score(y_test, y_pred)
    print('acc : ', score)
    
    print('Thresh =%.3f, n=%d, R2 : %.2f%%' %(thresh,select_x_train.shape[1],
                                              score*100))