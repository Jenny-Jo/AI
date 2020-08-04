import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import load_diabetes

dataset = load_diabetes()
x = dataset.data
y = dataset.target

print(x.shape)
print(y.shape)
'''
pca = PCA(n_components=5) # 10개 중 5개 컬럼으로 압축
x2 = pca.fit_transform((x))
pca_evr = pca.explained_variance_ratio_ # 압축한 컬럼(특성)들의 중요도순으로 5개 배열
print(pca_evr)
print(sum(pca_evr))
'''
pca = PCA()
pca.fit(x)
cumsum = np.cumsum(pca.explained_variance_ratio_)
print(cumsum)

aaa = np.argmax(cumsum >= 0.94 ) +1 # True, False로 나옴
print(cumsum >= 0.94)
print(aaa)

'''[0.40242142 0.55165324 0.67224947 0.76779711 0.83401567 0.89428759
 0.94794364 0.99131196 0.99914395 1.        ]
[False False False False False False  True  True  True  True]
7'''