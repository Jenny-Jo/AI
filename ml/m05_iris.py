# 다중분류 kneighbor// acc 확인하기

from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# 1. data
iris = load_iris()
x = iris.data
y = iris.target

print(x.shape, y.shape) # (150, 4) (150,)

print(x)
print(y)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(x)
x_scaled = scaler.transform(x)
print(x_scaled)

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca.fit(x_scaled)
x_pca = pca.transform(x_scaled)
print(x_pca)

###
from keras.utils import np_utils
y = np_utils.to_categorical(y)
###
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x_pca, y, random_state=66, shuffle=True,
    train_size = 0.8)

# 2. model
model = KNeighborsClassifier()

# 3. compile, fit
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["acc"])
model.fit(x_train, y_train, batch_size=30, epoch=30, validation_split=0.2)

# 4. evaluation, predict

loss,acc = model.evaluate(x_test, y_test, batch_size=100)

y_predict = model.predict(x_test)

y_test = np.argmax(y_test, axis=-1)
y_predict= np.argmax(y_predict, axis=-1)

print(f"loss:{loss}")
print(f"acc:{acc}")



