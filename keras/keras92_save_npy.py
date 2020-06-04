from sklearn.datasets import load_iris
import numpy as np

iris = load_iris()

print(type(iris)) #<class 'sklearn.utils.Bunch'>

x_data = iris.data
y_data = iris.target

print(x_data)
print(y_data)

np.save('./data/iris_x.npy', arr=x_data)
np.save('./data/iris_y.npy', arr=y_data)

x_data_load = np.load('./data/iris_x.npy')
y_data_load = np.load('./data/iris_y.npy')

print(type(x_data_load)) #<class 'numpy.ndarray'>
print(type(y_data_load))
print(x_data_load.shape) #(150,4)
print(y_data_load.shape) #(150,)

'''
1) 체크포인트/모델+가중치
fit 바로 전
modelpath = './model/sample/폴더명/ {epoch:02d} - {val_loss:  . 4f}_checkpoint_best.hdf5'
checkpoint = ModelCheckpoint(filepath = modelpath, monitor = 'val_loss', 
                            verbose =1,
                            save_best_only= True, save_weights_only= False)

callbacks에 불러주고/넣어주기

2) 모델+가중치
모델(model)이나 fit 뒤(model+W)에다가
model.save('./model/sample/폴더명/이름_model.save.h5')

3) 가중치
fit 다음
model.save_weights('./model/sample/폴더명/이름_save_weight.h5')
'''
