# copy & paste keras 101 , apply lr and tune.
# convert LSTM to Dense

# make optimizer & lr parameter 
# apply to random search

import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Input, Conv1D, MaxPooling1D, Flatten
from array import array

#1. Data
a = np.array(range(1,101))
size = 5                    #time_steps = 4 

def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1) : 
        subset = seq [ i: (i+size)]
        aaa.append(subset)
    return np.array(aaa)

dataset = split_x(a,size) #(96,5)

print(dataset.shape)
print(type(dataset)) #<class 'numpy.ndarray'>

# x = dataset[0:6,0:4]
# y = dataset[0:6,4:5]
x = dataset[ : , 0:4] #[모든 행, 0,1,2,3열 ]
y = dataset[:, 4]     #[모든 행, 인덱스 4]

print(x.shape) # (96, 4)
print(y.shape) # (96, )

#2.모델구성
model = Sequential()

model.add(Dense(100), input_shape=(4, 1))
model.add(Dense(100))
model.add(Dense(1))
model.summary()


#3.실행
from keras.losses import mse
from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss', patience=5, mode='auto')

### lr
weight = 0.5           # 초기 가중치
input = 0.5            # x
goal_prediction = 0.8  # y

lr = 0.001

for iteration in range(1101):                           # 0.5를 넣어서 0.8을 찾아가는 과정
    prediction = input*weight                           # y = w*x 
    error = (prediction - goal_prediction)**2           # loss

    print('Error : ' + str(error)+'\tPrediction : '+str(prediction))

    up_prediction = input *(weight + lr)                # weight = gradient : -경사 올림
    up_error = (goal_prediction - up_prediction)**2     # loss

    down_predicrion = input*(weight - lr)               # weight = gradient : +경사 내림
    down_error = (goal_prediction - down_predicrion)**2 # loss

    if(down_error < up_error):                          
        weight = weight - lr                            

    if(down_error > up_error):                          
        weight = weight + lr                


### make optimizer & lr parameter 

# hyper parameter가 2개나 늘었고, the most used hyperparameter is learning rate
from keras.optimizers import Adam, RMSprop, SGD, Adadelta, Adagrad, Nadam
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
optimizer = Adam(lr = 0.001)              # 0.013134053908288479, 3.45393
# optimizer = RMSprop(lr = 0.001)          #0.0013843015767633915, 3.4504988 이게 제일 좋은 듯
# optimizer = SGD(lr = 0.001)             # 0.08092157542705536,  3.3419425
# optimizer = Adadelta(lr = 0.001)        # 6.919477939605713 ,  0.13778175
# optimizer = Adagrad(lr = 0.001)        #  0.2491438090801239, 2.834703
# optimizer = Nadam(lr = 0.001)          #  0.3181155323982239,  3.1656237


model.compile (optimizer='adam', loss = 'mse', metrics = ['mse'])

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 13, train_size =0.8)

### apply to random search
parameters ={
    "n_estimators" : [100, 200],      
    "max_depth": [6, 8, 10, 20],      
    "min_samples_leaf":[3, 5, 7, 10], 
    "min_samples_split": [2,3, 5],     
    # 'max_features                   
    "n_jobs" : [-1]
}                                    

kfold = KFold(n_splits = 5, shuffle = True)                                                

model = RandomizedSearchCV(RandomForestClassifier(), parameters, cv =  kfold)  

model.fit(x_train, y_train)

print('최적의 매개변수 : ', model.best_estimator_)
y_pred = model.predict(x_test)
print("최종 정답률 = ", accuracy_score(y_test, y_pred) )
###
model.fit(x, y, epochs=1, batch_size =1 , verbose =1,
         callbacks = [es])
###

#4. 평가, 예측
loss, acc = model.evaluate(x, y)

y_predict = model.predict(x)
print('loss: ', loss)
print('mse:', mse)
print('y_predict: ', y_predict)
