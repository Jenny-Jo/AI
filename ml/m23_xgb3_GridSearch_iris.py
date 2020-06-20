from xgboost import XGBClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt


iris = load_iris()
x = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8,
                                                    shuffle = True, random_state = 66)


parameters = {
            "n_estimators":[90, 110, 200, 400], 
            "learning_rate": [0.1, 0.001,  0.5, 0.07, 0.05],
            "max_depth": [4, 5, 6, 7, 8],
            "colsample_bytree":[0.6, 0.9, 0.7, 1],
            "colsample_bylevel": [0.6, 0.7, 0.8, 0.9]
}

model = GridSearchCV(XGBClassifier(), parameters, cv =5, n_jobs = -1)

model.fit(x_train, y_train)

print("======================")
print(model.best_estimator_)
print('======================')
print(model.best_params_)
print('======================')

score = model.score(x_test, y_test)
print('점수 :', score)




'''
======================
XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.6, gamma=0, gpu_id=-1,
              importance_type='gain', interaction_constraints='',
              learning_rate=0.001, max_delta_step=0, max_depth=4,
              min_child_weight=1, missing=nan, monotone_constraints='()',
              n_estimators=90, n_jobs=0, num_parallel_tree=1,
              objective='multi:softprob', random_state=0, reg_alpha=0,
              reg_lambda=1, scale_pos_weight=None, subsample=1,
              tree_method='exact', validate_parameters=1, verbosity=None)
======================
{'colsample_bytree': 0.6, 'learning_rate': 0.001, 'max_depth': 4, 'n_estimators': 90} 
======================
점수 : 0.9666666666666667
'''