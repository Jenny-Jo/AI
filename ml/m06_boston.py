# 회귀 regressor
# 회귀 score, R2 비교
# 분류 score와 accuracy_score 비교


from sklearn.svm import LinearSVC, SVC #회귀 모델
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
import numpy as np

from sklearn.datasets import load_boston

dataset = load_boston()
x = dataset.data
y = dataset.target

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(x)
x_scaled = scaler.transform(x)
print(x_scaled)

from sklearn.decomposition import PCA

pca = PCA(n_components=2) #차원 두개로 나눠준다
pca.fit(x_scaled)
x_pca = pca.transform(x_scaled)
print(x_pca)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_pca, y, random_state= 66, 
shuffle= True, train_size = 0.8)

# 2. Model

# model = LinearSVC() # error
# model = SVC() # error
# model = KNeighborsClassifier(n_neighbors = 1) # error
model = KNeighborsRegressor(n_neighbors = 1) # 0.44
# model = RandomForestClassifier()  # error                    
# model = RandomForestRegressor() #이게 나음/ R2 0.68

# 3. Fit
model.fit(x_train, y_train)

# 4. score, predict
from sklearn.metrics import accuracy_score, r2_score
y_predict = model.predict(x_test)

score = model.score(x_test, y_test)
print('score', score)

# acc = accuracy_score(y_test, y_predict)
R2 = r2_score(y_test, y_predict)
print('R2:', R2)
# print('acc', acc)
