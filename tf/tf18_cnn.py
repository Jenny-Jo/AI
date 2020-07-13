import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split as tts
from keras.datasets import mnist

# 데이터 입력
# dataset = load_iris()
(x_train,y_train),(x_test,y_test)=mnist.load_data()

print(x_train.shape)#(60000, 28, 28) > 4차원으로 
print(y_train.shape)#(60000,)


x_train = x_train.reshape(-1,x_train.shape[1],x_train.shape[2],1).astype('float32')/255
x_test = x_test.reshape(-1,x_test.shape[1],x_test.shape[2],1).astype('float32')/255

from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

learning_rate = 0.001
training_epochs = 15
batch_size = 100
total_batch = int(len(x_train)/batch_size) # 60000/100

x = tf.placeholder(tf.float32,[None, 784])
x_img = tf.reshape(x, [-1, 28, 28, 1])
y = tf.placeholder(tf.float32,[None, 10])

keep_prob = tf.placeholder(tf.float32) # dropout

# 1st hidden layer
# w1 = tf.Variable(tf.random_normal[784, 512], name ='weight') # 

w1 = tf.get_variable('w1',shape=[3, 3, 1, 32]) 
# conv2D(32 output, (3,3)kernel size, input_shape=(28,28,1))
# [3, 3, kernel size,/ 1, color,/ 32 output]

print('w1:', w1) 
# w1: <tf.Variable 'w1:0' shape=(3, 3, 1, 32) dtype=float32_ref>

layer1 = tf.nn.conv2d(x_img, w1, strides=[1,1,1,1], padding='SAME') # ?
print('layer:',layer1) 
# layer: Tensor("Conv2D:0", shape=(?, 28, 28, 32), dtype=float32)

layer1 = tf.nn.selu(layer1)
layer1 = tf.nn.max_pool(layer1,ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME') # kernel size
print('layer:',layer1)
# layer: Tensor("MaxPool:0", shape=(?, 14, 14, 32), dtype=float32)


                             #kernel  #input #output
w2 = tf.get_variable('w2',shape=[3, 3, 32, 64]) # variable보다 좋다
layer2 = tf.nn.conv2d(layer1, w2, strides=[1,1,1,1], padding='SAME') # ?
layer2 = tf.nn.selu(layer2)
layer2 = tf.nn.max_pool(layer2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
print(layer2) # (?, 7, 7, 64)

layer_flat = tf.reshape(layer2,[-1, 7*7*64])


w3 = tf.get_variable('w3', shape=[7*7*64,10], initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([10]))
hypothesis = tf.nn.softmax(tf.matmul(layer_flat, w3)+b3)



b2 = tf.Variable(tf.random_normal([512]))
layer2 = tf.nn.selu(tf.matmul(layer1,w2)+b2)
layer2 = tf.nn.dropout(layer2, keep_prob=keep_prob)

w3 = tf.get_variable('w3',shape=[512, 512],initializer=tf.contrib.layers.xavier_initializer()) # variable보다 좋다
b3 = tf.Variable(tf.random_normal([512]))
layer3 = tf.nn.selu(tf.matmul(layer2,w3)+b3)
layer3 = tf.nn.dropout(layer3, keep_prob=keep_prob)


w4 = tf.get_variable('w4',shape=[512, 256],initializer=tf.contrib.layers.xavier_initializer()) # variable보다 좋다
b4 = tf.Variable(tf.random_normal([256]))
layer4 = tf.nn.selu(tf.matmul(layer3,w4)+b4)
layer4 = tf.nn.dropout(layer4, keep_prob=keep_prob)

w5 = tf.get_variable('w5',shape=[256, 10],initializer=tf.contrib.layers.xavier_initializer()) # variable보다 좋다
b5 = tf.Variable(tf.random_normal([10]))
layer5 = tf.Variable(tf.random_normal([10]))
hypothesis = tf.nn.softmax(tf.matmul(layer4,w5) + b5)

cost = tf.reduce_mean(-tf.reduce_sum(y* tf.log(hypothesis), axis=1))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for epoch in range(training_epochs): # 15
    avg_cost = 0
    for i in range(total_batch): #600
        ################################################
   
        ''''''
        batch_xs, batch_ys = x_train[0:100], y_train[0:100]
        batch_xs, batch_ys = x_train[100:200], y_train[0:100]
                                     
        batch_xs, batch_ys = x_train[i *100 : (i+1)*100], y_train[i*100:(i+1)*100]  # batch_size = 100
        ''''''
        start = i * batch_size 
        end = start + batch_size                                            
        batch_xs, batch_ys = x_train[start:end], y_train[start:end]
        
        # weight 값에 대해선 dropout 적용 안된다
        ##################################################
        feed_dict = {x:batch_xs, y:batch_ys, keep_prob:0.7}
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c/total_batch
    print('Epoch: ', '%04d'%(epoch + 1), 'cost= {:.9f}'.format(avg_cost))
print('훈련끝')

prediction = tf.equal(tf.arg_max(hypothesis,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
print('acc: ', sess.run(accuracy, feed_dict={x:x_test, y:y_test, keep_prob:1}))
