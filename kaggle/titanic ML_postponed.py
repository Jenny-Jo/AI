import pandas as pd
import numpy as np
from keras.layers import Dense, LSTM, Dropout
from keras.models import Sequential

# 1. data 
train = pd.read_csv('kaggle/data/train.csv', sep=',')
test = pd.read_csv('kaggle/data/test.csv',sep=',')
gender_df = pd.read_csv('kaggle/data/gender_submission.csv',header=0, sep=',')

# 2.EDA (Exploratory data analysis)
print(train.head(5))
print(train.info()) # (891* 12) / 
print(test.head()) # survived feature 없음
print(test.info()) # (418*11) / 
print(train.isnull().sum()) # age 117, cabin 667 ,embarked 2 에 null값 존재 
print(test.isnull().sum()) # age 86, fare 1, cabin 327, 에 null값 존재

# 3. visualization
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# 3-1. Bar Chart
#Pclass
#Sex
#SibSp
#Parch
#Embarked
#Cabin

def bar_chart(feature):
    survived = train[train['Survived']==1][feature].value_counts()
    dead = train[train['Survived']==0][feature].value_counts()
    df = pd.DataFrame([survived,dead])
    df.index = ['Survived','Dead']
    df.plot(kind='bar',stacked=True, figsize=(10,5))
    plt.title(feature, fontsize = 30)
    plt.show()
# bar_chart('Sex') # 여자가 더 많이 살았다
# bar_chart('Pclass') # 3 class가 많이 죽음, 1 class는 상대적으로 더 삼
# bar_chart('SibSp') # single 일 수록 많이 죽고 ...?
# bar_chart('Parch') # 부모가 있으면 그나마 살 확률이 늘어난다
# bar_chart('Embarked') # S가 압도적으로 수가 일단 많다. C면 반반 인 듯
# bar_chart('Cabin') # 분석하기 힘듬

# 4. feature engineering
# 4-1. 타이타닉 침몰 경위
# Img(url= 'https://camo.g/thubusercontent.com/e15a5414be97decd975cfed68e0c0f79e768f7e7/68747470733a2f2f737461746963312e73717561726573706163652e636f6d2f7374617469632f3530303634353366653462303965663232353262613036382f742f3530393062323439653462303437626135346466643235382f313335313636303131333137352f544974616e69632d537572766976616c2d496e666f677261706869632e6a70673f666f726d61743d3135303077')
# 상대적으로 배 밑부분에 있던 3등석 승객들이 많이 사망함

# 4-2 Name - 여성의 결혼여부(Miss, Mrs)로 인해 자식이 있음으로 인해 생존 확률이 더 높을 수 있음
train_test_data = [train, test] 
for dataset in train_test_data:
    dataset['Title']= dataset['Name'].str.extract('([A-Za-z]+)\.', expand=False)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              

print(train['Title'].value_counts())

print(test['Title'].value_counts())

# 4-2-1 Title map

# 간단하게 만드는 방법?
'''
serviceperson = ['Master','Col','Major','Capt'] # 군인 # 3
Mr = ['Dr','Rev','Don','Sir','Jonkheer'] # 0
Miss = ['Mlle', 'Lady'] # 1
Mrs = ['Mme', 'Countess','Dona'] # 2

title_mapping = { Mr :0 , Miss:1, Mrs:2, serviceperson:3 }'''

# 일일이 딕셔너리 쳐주는거 너무 귀찮은데 어떻게 간단하게 안될까???
title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2, 
                 "Master": 3, "Dr": 0, "Rev": 0, "Col": 3, "Major": 3, "Mlle": 1,"Countess": 2,
                 "Ms": 1, "Lady": 1, "Jonkheer": 0, "Don": 0, "Dona" : 2 , "Mme": 2,"Capt": 3,"Sir": 0 }

for dataset in train_test_data:
    dataset['Title'] = dataset['Title'].map(title_mapping)

print(train.head())
print(test.head())
bar_chart('Title')
('==========================')
# Delete unnecessary feature from dataset
train = train.drop('Name', axis=1, inplace=True)
test = test.drop('Name', axis=1, inplace=True)

print(train.head())
print(test.head())