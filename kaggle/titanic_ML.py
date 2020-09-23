import pandas as pd
import numpy as np
from keras.layers import Dense, LSTM, Dropout
from keras.models import Sequential

# 1. data 2. analysis
train = pd.read_csv('kaggle/data/train.csv', sep=',')
test = pd.read_csv('kaggle/data/test.csv',sep=',')
# print(train.head(5)) # colomn survived  있는 data
# print(test.head(5))  # 데이타의 모양

# # 데이타 열, 타입과 결측치 여부
# print(train.info()) # 891 rows * 12 colomns, ㅇ
# print(test.info())  # 418 rows * 11 colomns


# print(train.isnull().sum()) # 결측치
# print(test.isnull().sum())


# print(train.describe()) # 컬럼별 통계적 수치
# print(test.describe())

# Visualization

# 1)pie chart
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
sns.set() 

def pie_chart(feature):
    feature_ratio = train[feature].value_counts(sort=False)
    feature_size = feature_ratio.size
    feature_index = feature_ratio.index
    survived = train[train['Survived']==1][feature].value_counts()
    dead = train[train['Survived']==0][feature].value_counts()
    
    plt.plot(aspect='auto')
    plt.pie(feature_ratio, labels=feature_index, autopct='%1.1f%%')
    plt.title(feature + '\'s ratio in total')
    plt.show()

    for i , index in enumerate(feature_index):
        plt.subplot(1, feature_size + 1, i +1, aspect='equal')
        plt.pie([survived[index], dead[index]], labels=['Survived', 'Dead'], autopct='%1.1f%%')
        plt.title(str(index) + '\'s ratio in total')
    plt.show()

# pie_chart('Sex')
# pie_chart('Pclass')
# pie_chart('Embarked')


# 2) bar chart
def bar_chart(feature):
    survived = train[train['Survived']==1][feature].value_counts()
    dead = train[train['Survived']==0][feature].value_counts()
    df = pd.DataFrame([survived,dead])
    df.index = ['Survived','Dead']
    df.plot(kind='bar',stacked=True, figsize=(10,5))
    plt.show()

# bar_chart('Parch')    
# bar_chart('SibSp')

# 3. feature engineering
## 3-1. name feature

train_test_data = [train, test]
for dataset in train_test_data:
    dataset['Title']= dataset.Name.str.extract('([A-Za-z]+)\.')

print(train.head(5))
print(pd.crosstab(train['Title'], train['Sex']))

# replace Title
'''
serviceperson = ['Master','Col','Major','Capt'] # 군인 # 3
Mr = ['Dr','Rev','Don','Sir','Jonkheer'] # 0
Miss = ['Mlle', 'Lady'] # 1
Mrs = ['Mme', 'Countess','Dona'] # 2'''


for dataset in train_test_data:
    dataset['Title']= dataset['Title'].replace(['Master','Col','Major','Capt'], 'serviceperson')
    dataset['Title']= dataset['Title'].replace(['Dr','Rev','Don','Sir','Jonkheer'], 'Mr')
    dataset['Title']= dataset['Title'].replace(['Mlle', 'Lady'], 'Miss')
    dataset['Title']= dataset['Title'].replace(['Mme', 'Countess','Dona'], 'Mrs')

'''    dataset['Title']= dataset['Title'].replace(['Capt', 'Col', 'Don','Dona','Dr','Countess', 'Jonkheer', 'Major', 'Rev','Sir'], 'Other')
    dataset['Title']= dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title']= dataset['Title'].replace('Mme','Mrs')
    dataset['Title']= dataset['Title'].replace('Ms', 'Miss')
    dataset['Title']= dataset['Title'].replace('Lady', 'Miss')'''

print(train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())



# convert the categorical Title value into numeric value
for dataset in train_test_data:
    dataset['Title'] = dataset['Title'].astype(str)
print(train.head(5))

## 3-2. sex feature- convert the categorical Sex value into numeric value
for dataset in train_test_data:
    dataset['Sex'] = dataset['Sex'].astype(str) # 왜 문자열로 바꿔주는거지??
    
print(train.Embarked.unique())
print(train.Embarked.value_counts(dropna=False))

## 3-3. Embarked Feature
# Get the NaN into 'S', because the 'S' category has the biggest value

print(train.Embarked.unique())
print(train.Embarked.value_counts(dropna=False))

for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')

print(train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean())

for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].astype(str)
    
## 3-4. Age Feature
# Fill the Nan value with random number of Same Title, and combine similar age by group

for dataset in train_test_data:
    dataset['Age'].fillna(dataset['Age'].mean(), inplace=True)
    print(dataset['Age'].isnull().sum())
    dataset['Age'] = dataset['Age'].astype(int)
    train['AgeBand'] = pd.cut(train['Age'], 5)
print(train[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean())

print(train.head())

for dataset in train_test_data:
    dataset.loc[dataset['Age']<=16, 'Age'] = 0
    dataset.loc[(dataset['Age']>16)&(dataset['Age']<=32), 'Age']=1
    dataset.loc[(dataset['Age']>32)&(dataset['Age']<=48), 'Age']=2
    dataset.loc[(dataset['Age']>48)&(dataset['Age']<=64), 'Age']=3
    dataset.loc[dataset['Age']>64, 'Age'] = 4
    dataset['Age']=dataset['Age'].map({0:'child', 1:'Young', 2:'Middle',3:'Prime', 4:'Old'}).astype(str) # 왜 for문 안에 들어가야 하는 것인가?
print(train.head())

## 3-5. Fare Feature
print(train[['Pclass', 'Fare']].groupby(['Pclass'], as_index=False).mean())

for dataset in train_test_data:
    dataset['Fare'] = dataset['Fare'].fillna(13.675) # the only one emtpy Pcalss'es fare

train['FareBand']= pd.qcut(train['Fare'],5)
print(train[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean())

for dataset in train_test_data: # train_test로 구분한 이유가 뭘까??
    dataset.loc[dataset['Fare']<=7.854, 'Fare']=0 # loc 뭐였더라?
    dataset.loc[(dataset['Fare']>7854)&(dataset['Fare']<=10.5), 'Fare']=1
    dataset.loc[(dataset['Fare']>10.5)&(dataset['Fare']<=21.679), 'Fare']=2
    dataset.loc[(dataset['Fare']>21.679)&(dataset['Fare']<=39.688), 'Fare']=3
    dataset.loc[dataset['Fare']>39.688, 'Fare']=4
    dataset['Fare']=dataset['Fare'].astype(int)
    
# 3.6. Family feature
for dataset in train_test_data:
    dataset['Family'] = dataset['Parch'] + dataset['SibSp']
    dataset['Family'] = dataset['Family'].astype(int)

# 3.7 Feature selection
features_drop=['Name', 'Ticket', 'Cabin', 'SibSp', 'Parch']
train = train.drop(features_drop, axis=1)
test = test.drop(features_drop, axis=1)
train = train.drop(['PassengerId', 'AgeBand', 'FareBand'], axis=1)


# one-hot-encoding for categorical variable ?? 이해가 안됨. 
train = pd.get_dummies(train)
test = pd.get_dummies(test)

train_data = train.drop('Survived', axis=1) # x값
train_label = train['Survived'] # y값
test_data = test.drop('PassengerId', axis=1).copy() #.copy를 넣은 이유?/ x_test
# gender_df = gender_df.drop('PassengerId', axis=1).copy()#y_test // 내 최대의 실수!!!!!!!!!!!!!!!!!!! 이건 그냥 서밋 예제였을 뿐ㅠㅠ 공홈 안본 내 잘못이다!

print(train_data.shape, train_label.shape, test_data.shape)
# (891, 18) (891,) (418, 18) 
# x_train, y_train, x_test, y_test

# 4. Modeling & Testing
'''
Logistic Regression
SVM
kNN
Random Forest
Navie Bayes
ㅁ 여기에 그리드서치나 랜덤서치 추가할 것
ㅁ 다른 모델 XGB 같은거 추가'''


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import RandomizedSearchCV # 새로 추가한 것

from sklearn.utils import shuffle
import xgboost

# shuffle
train_data, train_label = shuffle(train_data, train_label, random_state = 5)

# pipeline
def train_and_test(model) : 
    model.fit(train_data, train_label)
    prediction = model.predict(test_data)
    accuracy = round(model.score(train_data, train_label)* 100, 2)
    print ( 'Accuracy: ', accuracy, '%')
    '''
    # 1) randomized search
    # https://www.kaggle.com/simulacra/titanic-with-xgboost, m12 볼 것

    model_param_grid = {
        'n_estimators':range(8,20),
        'max_depth':range(6,10),
        'learning_rate':[.4, .45, .5, .55, .6],
        'colsample_bytree':[.6, .7, .8, .9,1]
    }
    random_search = RandomizedSearchCV(param_distributions=model_param_grid, 
                                           estimator = model, scoring='accuracy',
                                           verbose=1, n_iter=50, cv=4)
    random_search.fit(train_data, train_label)
    print('Best Parameters found: ', random_search.best_params_)
    print('Best accuracy found: ', random_search.best_score_)
        
    # 2) feature importance m18 보기
    # feat_importances=pd.Series(model.feature_importances_, index=train_data.columns)
    # feat_importances.nlargest(10).plot(kind='barh')
    # plt.show()
    return prediction
# model'''

# 각 모델별로 돌리기
# Logistic Regression
log_pred = train_and_test(LogisticRegression())
# SVM
svm_pred = train_and_test(SVC())
#Random Forest
rf_pred = train_and_test(RandomForestClassifier(n_estimators=100))
# Naive Bayes
nb_pred = train_and_test(GaussianNB())
# XGB
xgb_pred = train_and_test(XGBClassifier())
# xgboost.plot_importance(XGBClassifier)
# 여기 안에 이름 넣어서 출력 하고 싶은데.... 어떻게 하지?
'''
Accuracy:  82.72 %
Accuracy:  83.39 %
Accuracy:  88.33 % randomforest
Accuracy:  82.49 %
Accuracy:  87.99 %'''

print(test.head())
submission = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived': rf_pred
})

submission.to_csv('kaggle\submission/sumbission_88_rf.csv', index=False)
# pd.read_csv('kaggle\submission/sumbission_88_rf.csv',sep=',')