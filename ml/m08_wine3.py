import pandas as pd
import matplotlib.pyplot as plt

# 1. Data
wine = pd.read_csv('./ml/winequality-white.csv', sep=';', header=0)

count_data = wine.groupby('quality')['quality'].count()

print(count_data)

count_data.plot()
plt.show()
# quality
# 3      20
# 4     163
# 5    1457
# 6    2198
# 7     880
# 8     175
# 9       5
# Name: quality, dtype: int64