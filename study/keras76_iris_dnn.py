from sklearn.datasets import load_iris

iris = load_iris()
x = iris.data
y = iris.target

print(x.shape, y.shape) #(150, 4) (150,)

###
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(x)
x_sclaed = scaler.transform(x)
print(x_sclaed)

from sklearn.decomposition import PCA