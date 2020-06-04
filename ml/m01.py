import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 10, 0.1) # x가 0.1 씩 증가
y = np.sin(x)

plt.plot(x, y)

plt.show()


