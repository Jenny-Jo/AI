import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(0, x)

x = np.arange(-5, 5, 0.1)
y = relu(x)

plt.plot(x, y)
plt.grid()
plt.show()

# 0 이하는 소멸, Leaky relu> elu> selu 차례로 단점 보완하며 나옴
# relu, leaky relu, elu, selu순으로 인기 많아   

                    
import numpy as np
def elu(x):
    if(x>0):
        return x
    if(x<0):
        return 0.2*(np.exp(-x)-1)

def elu(x):
    y_list = []
    for x in x:
        if(x>0):
            y = x
        if(x<0):
            y = 0.2*(np.exp(x)-1)
        y_list.append(y)
    return y_list

def elu(x):
    x = np.copy(x)
    x[x<0]=0.2*(np.exp(x[x<0])-1)
    return x


def leakyrelu(x):                      
    return np.maximum(0.01*x, x)        

def elu(x):                 # for문 쓴거
    y_list = []
    for x in x:
        if(x>0):
            y = x
        if(x<0):
            y = 1*(np.exp(x)-1)
        y_list.append(y)
    return y_list

def elu(x):                  # lamda 쓴거
    return list(map(lambda x : x if x > 0 else 1*(np.exp(x)-1), x))

def selu(x):                 # for문 쓴거
    y_list = [] 
    for x in x:
        if x > 0:
            y = x
        if x <= 0:
            y = 1.67326*(np.exp(x) - 1)
        y_list.append(y)
    return y_list

def selu(x, a = 1.6732):  # lambda쓴거
    return list(map(lambda x : x if x > 0 else a*(np.exp(x)-1), x))