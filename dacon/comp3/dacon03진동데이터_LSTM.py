import numpy as np
import pandas as pd
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 1. data

# 1) 불러 오기
test_features = pd.read_csv("./data/dacon/comp2/test_features.csv",  header = 0, index_col = 0)
train_features = pd.read_csv("./data/dacon/comp2/train_features.csv", header = 0, index_col = 0)
train_target = pd.read_csv("./data/dacon/comp2/train_target.csv", header = 0, index_col = 0)

# 2) shape
print(train_features)   # (1050000, 5)
print(test_features)    # (262500, 5)
print(train_target)     # (2800, 4)









