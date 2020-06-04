# 이중분류


from sklearn.svm import LinearSVC, SVC #회귀 모델
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
import numpy as np

from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
dataset = load_breast_cancer()
x = cancer.data
y = cancer.target
print('x', x)
print('y', y )
scaler = MinMaxScaler()
scaler.fit(x)
x_scaled = scaler.transform(x)
print(x_scaled)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, random_state=66 )

# 2. model

# model = LinearSVC()
# model = SVC()
# model = KNeighborsClassifier(n_neighbors = 1) 
# model = KNeighborsRegressor(n_neighbors = 1)
# model = RandomForestClassifier()
model = RandomForestRegressor()

# 3. fit
model.fit(x_train, y_train)

# 4. score, predict
from sklearn.metrics import accuracy_score, r2_score
y_predict = model.predict(x_test)

score = model.score(x_test, y_test)
print('score:', score)

acc = accuracy_score(y_test, y_predict)
print('acc: ', acc)

R2 = r2_score(y_test, y_predict)
print('R2: ', R2)

# 분류
# score: 0.9790209790209791
# acc:  0.9790209790209791
# R2:  0.9085677749360613

# 분류
# score: 0.986013986013986
# acc:  0.986013986013986
# R2:  0.9390451832907076

# 분류
# score: 0.9440559440559441
# acc:  0.9440559440559441
# R2:  0.7561807331628303

# 회귀
# score: 0.7561807331628303
# acc:  0.9440559440559441
# R2:  0.7561807331628303

# 분류
# score: 0.958041958041958
# acc:  0.958041958041958
# R2:  0.8171355498721227

# 회귀 ValueError: Classification metrics can't handle a mix of binary and continuous targetsr