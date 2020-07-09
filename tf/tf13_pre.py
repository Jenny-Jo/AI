# preprocessing

import tensorflow as tf
import numpy as np

def min_max_scaler(dataset):
    numerator = dataset - np.min(dataset, 0)  # 0 : 열에서 최소값을 찾겠다. / 809.51
    denominator = np.max(dataset, 0) -  np.min(dataset, 0)
                        # 0 : 열에서 최대값 / 828
    return numerator / (denominator + 1e-7)  


dataset = np.array(

    [

        [828.659973, 833.450012, 908100, 828.349976, 831.659973],

        [823.02002, 828.070007, 1828100, 821.655029, 828.070007],

        [819.929993, 824.400024, 1438100, 818.97998, 824.159973],

        [816, 820.958984, 1008100, 815.48999, 819.23999],

        [819.359985, 823, 1188100, 818.469971, 818.97998],

        [819, 823, 1198100, 816, 820.450012],

        [811.700012, 815.25, 1098100, 809.780029, 813.669983],

        [809.51001, 816.659973, 1398100, 804.539978, 809.559998],

    ]

)


dataset = min_max_scaler(dataset)
print(dataset)

x_data = dataset[:, 0:-1]
y_data = dataset[:,[-1]]

print(x_data.shape, y_data.shape) # (8, 4) (8, 1)

# 회귀
# x, y , w, b, hypothesis, cost, train(optimizzer)

