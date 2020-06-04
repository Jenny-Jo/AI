import numpy as np
import pandas as pd

samsung = pd.read_csv('./test_samsung.py/samsung_stock.csv', header=0, index_col=0,sep=',')
hite = pd.read_csv('./test_samsung.py/hite_stock.csv', header=0, index_col=0, sep=',')

print(samsung)

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

x_s,y_s=split1(samsung_array,5)
x_h,y_h=split1(hite_array,5)

    # print(f"x_h.shape:{x_h.shape}")
    # print(f"x_s.shape:{x_s.shape}")

    #scaler 위해서 reshape
    #했으나, 어펜드 하면서 취소.
    # x_s= x_s.reshape(-1,x_s.shape[1]*x_s.shape[2])
    # x_h= x_h.reshape(-1,x_h.shape[1]*x_h.shape[2])
