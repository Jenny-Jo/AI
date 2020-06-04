# 회귀 regressor

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

pca = PCA(n_components=2)
pca.fit(x_scaled)
x_pca = pca. transform(x_scaled)
print(x_pca)

from sklearn.model_selection import train_test_split

