import numpy as np
import matplotlib.pyplot as plt

f = lambda x : x**2 - 4*x + 6
x = np.linspace(-1, 6, 100) # 이거 뭐지
y = f(x)

# 그림 그리기
plt.plot(x, y, 'k-') 
plt.plot(2, 2, 'sk') # sk 공부
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.show()

gradient = lambda x : 2*x - 4 # f 함수 미분한 것 # 2차 함수에서 gradient가 가로로 일직선이 되는 0이 되는 지점에서 최적의 가중치를 구할 수 있음
x0 = 0.0
MaxIter = 500
learning_rate = 0.1

print('step\tx\tf(x)')
print('{:02d}\t{:6.5f}\t{:6.5f}'.format(0, x0, f(x0)))

for i in range(MaxIter):
    x1 = x0 - learning_rate * gradient(x0)
    x0 = x1
    
    print('{:02d}\t{:6.5f}\t{:6.5f}'.format(i+1, x0, f(x0)))
    

