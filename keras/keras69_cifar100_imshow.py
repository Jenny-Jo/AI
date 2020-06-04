from keras.datasets import cifar100
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = cifar100.load_data()
#colomn 백개 예제
#체크포인트, 시각화 텐서보드, cnn, dnn, lstm 

print(x_train[0])
print('y_train[0] : ', y_train[0])

print(x_train.shape) # (50000, 32, 32, 3)
print(x_test.shape)  # (10000, 32, 32, 3)
print(y_train.shape) # (50000, 1)
print(y_test.shape)  # (10000, 1)

plt.imshow(x_train[0])
plt.show()