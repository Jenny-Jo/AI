import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import load_diabetes

dataset = load_diabetes()
x = dataset.data
y = dataset.target

print(x.shape)
print(y.shape)

pca = PCA(n_components=5) # 10개 중 5개 컬럼으로 압축
x2 = pca.fit_transform((x))
pca_evr = pca.explained_variance_ratio_ # 압축한 컬럼(특성)들의 중요도순으로 5개 배열
print(pca_evr)
print(sum(pca_evr))
'''
(442, 10)
(442,)
[0.40242142 0.14923182 0.12059623 0.09554764 0.06621856]
0.8340156689459766

0.17 손해 봐도  83프로만 가져가도 충분히 가능'''