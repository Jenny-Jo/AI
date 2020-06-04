### 1)Data
from sklearn.datasets import load_iris

iris = load_iris()
x = iris.data
y = iris.target

print(x.shape, y.shape) #(150, 4) (150,)

print(x)
print(y)
###
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

print(x_train.shape) #(120, 2)
print(x_test.shape)  #(30, 2)
print(y_train.shape) #(120,)
print(y_test.shape)  #(30,)

### 2) Model
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()

model.add(Dense(10, input_shape = (2, )))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='relu'))


### 3) compile, fit

model.compile(loss='crossentropy', optimizer='adam', metrics=['acc'])

from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
##############
hist = model.fit(x_train, y_train, epochs=10, verbose=1, callbacks=[])

### 4) evaluate, predict



