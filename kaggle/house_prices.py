# https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings 
warnings.filterwarnings('ignore')
# %matplotlib inline


# 1. data
train = pd.read_csv('kaggle\data/train (1).csv')
test = pd.read_csv('kaggle\data/test (1).csv')

print(train.columns)
# print(train.head(5))
# print(test.head(5))
# print(train.info()) # 1460 * 81
# print(test.info())  # 1459 * 80
# # print(train.describe())
# # print(test.describe())
print(test.isnull().sum())
print(train.isnull().sum())

print(train['SalePrice'].describe())

sns.distplot(train['SalePrice']);
plt.show()

# 1. 
print('skewness: %f'%train['SalePrice'].skew())
print('Kurtosis: %f'%train['SalePrice'].kurt())
# skewness: 1.882876
# Kurtosis: 6.536282
# 한 쪽으로 기울어짐

# 2. 
# 2.1 Relationship with numerical variables
def scatter(var):
    data = pd.concat([train['SalePrice'], train[var]], axis = 1)
    data. plot.scatter(x=var, y ='SalePrice', ylim=(0.800000))
    plt.show()# 역시 함수로 만드니 편하구나!

scatter('GrLivArea') # linear relationship 
scatter('TotalBsmtSF') # 너무 한뭉탱이로 뭉친 듯?

# 2.2 Relationship with categorical features
var = 'OverallQual'
data = pd.concat([train['SalePrice'],train[var]], axis=1)
f,ax = plt.subplots(figsize=(8,6))
fig = sns.boxplot(x=var, y='SalePrice', data=data)
fig.axis(ymin=0, ymax=800000)
plt.show()

var = 'YearBuilt'
data = pd.concat([train['SalePrice'], train[var]], axis=1)
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x=var, y='SalePrice', data = data)
fig.axis(ymin=0, ymax=800000);
plt.xticks(rotation=90);
plt.show()

# 3. correlation  matrix( heatmap style)
corrmat = train.corr()
f, ax = plt.subplots(figsize = (12, 9))
sns.heatmap(corrmat, vmax=.8, square= True);
plt.show() # portch, pool area, MiscVal, MoSold, Yrsold, kitchen above Gr가 주요한 원인

# 3.1 saleprice correlation matrix
k = 10
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, 
                 fmt='.2f', annot_kws={'size':10}, yticklabels=cols.values, 
                 xticklabels=cols.values)
plt.show()

#scatterplot
sns.set()
cols = ['SalePrice','OverallQual','GrLivArea','GarageCars','TotalBsmtSF','FullBath','YearBuilt']
sns.pairplot(train[cols], size=1.0)
plt.show()

# missing data
total = train.isnull().sum().sort_values(ascending=False)
percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
print(missing_data.head(20))