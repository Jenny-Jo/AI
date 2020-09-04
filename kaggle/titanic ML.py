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
    df = pd.DataFrame(survived,dead)
    df.index = ['Survived', 'Dead']
    df.plot(kind='bar', stacked = True, figsize=(10,5))
    
print(bar_chart('Sex'))
print(bar_chart('Pclass'))
print(bar_chart('SibSp'))
print(bar_chart('Parch'))
print(bar_chart('Embarked'))
print(bar_chart('Cabin'))
