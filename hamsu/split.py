import numpy as np

def split_x(seq, size) :
    aaa = []
    for i in range(len(seq) - size +1):
        subset = seq [i : (i+size)]
        # aaa.append([item for item in subset])
        aaa.append(subset)
# dataset = split_x(a,size)에서 함수가 재사용 되기 때문에 seq = a 가 됨
    print(type(aaa))
    return np.array(aaa)



def split1(datasets,timesteps):#samsung
        x_values=list()
        y_values=list()
        for i in range(len(datasets)-timesteps+1):#10-5+1
            x=datasets[i:i+timesteps]
            x=np.append(x,datasets[i+timesteps,0])
            y=datasets[i+timesteps+1]
            x_values.append(x)
            y_values.append(y)
        return np.array(x_values),np.array(y_values)
