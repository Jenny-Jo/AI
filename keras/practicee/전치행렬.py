import numpy as np 
x = np.array([range(1,51), range(101,151), range (151,201)])
y = np.array(range(50, 101))

print(x.shape)

x= np.transpose(x)
x= np.array(x).T
x= np.swapaxes(x,0,1)
x=np.array(x).reshape(50,3)
print(x.shape)

 
