import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split as tts
from keras.datasets import mnist

# 데이터 입력
# dataset = load_iris()
(x_train,y_train),(x_test,y_test)=mnist.load_data()

print(x_train.shape)#(60000, 28, 28)
print(y_train.shape)#(60000,)


x_train = x_train.reshape(-1,x_train.shape[1]*x_train.shape[2])
x_test = x_test.reshape(-1,x_test.shape[1]*x_test.shape[2])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    y_train = sess.run(tf.one_hot(y_train,10))
    y_test = sess.run(tf.one_hot(y_test,10))
y_train=y_train.reshape(-1,10)
y_test=y_test.reshape(-1,10)

######### relu, selu, elu 하기 ####
w2 = tf.Variable(tf.zeros([100,50]),name='weight1')
b2 = tf.Variable(tf.zeros([50]), name='bias')
# model.add(Dense(50))
layer2 = tf.nn.selu(tf.matmul(layer1, w2)+b2)
layer2 = tf.nn.dropout(layer2, keep_prob=0.3)
######################################
# y_data = y_data.reshape(y_data.shape[0],1)

x = tf.placeholder(tf.float32, shape=[None,28*28])
y = tf.placeholder(tf.float32, shape=[None,10])

w = tf.Variable(tf.zeros([28*28,100]),name="weight")
b = tf.Variable(tf.zeros([100]),name="bias")
# layer = tf.nn.softmax(tf.matmul(x,w)+b)
layer = tf.nn.relu(tf.matmul(x, w2) + b2)
# layer = tf.sigmoid(tf.matmul(layer, w2) + b2)
# layer = tf.nn.elu(tf.matmul(layer, w2) + b2)
# layer = tf.nn.selu(tf.matmul(layer, w2) + b2)
#model.add(Dense(100,input_shape=(2,)))

w = tf.Variable(tf.zeros([100,50]),name="weight")
b = tf.Variable(tf.zeros([50]),name="bias")
# layer = tf.nn.softmax(tf.matmul(layer,w)+b)
layer = tf.nn.relu(tf.matmul(layer, w2) + b2)
# layer = tf.sigmoid(tf.matmul(layer, w2) + b2)
# layer = tf.nn.elu(tf.matmul(layer, w2) + b2)
# layer = tf.nn.selu(tf.matmul(layer, w2) + b2)
# model.add(Dense(50))

w = tf.Variable(tf.zeros([50,50]),name="weight")
b = tf.Variable(tf.zeros([50]),name="bias")
layer = tf.nn.softmax(tf.matmul(layer,w)+b)
# model.add(Dense(50))
layer = tf.nn.relu(tf.matmul(layer, w2) + b2)
# layer = tf.sigmoid(tf.matmul(layer, w2) + b2)
# layer = tf.nn.elu(tf.matmul(layer, w2) + b2)
# layer = tf.nn.selu(tf.matmul(layer, w2) + b2)


w = tf.Variable(tf.zeros([50,50]),name="weight")
b = tf.Variable(tf.zeros([50]),name="bias")
layer = tf.nn.softmax(tf.matmul(layer,w)+b)
# model.add(Dense(50))
layer = tf.nn.relu(tf.matmul(layer, w2) + b2)
# layer = tf.sigmoid(tf.matmul(layer, w2) + b2)
# layer = tf.nn.elu(tf.matmul(layer, w2) + b2)
# layer = tf.nn.selu(tf.matmul(layer, w2) + b2)


w = tf.Variable(tf.zeros([50,50]),name="weight")
b = tf.Variable(tf.zeros([50]),name="bias")
layer = tf.nn.softmax(tf.matmul(layer,w)+b)
# model.add(Dense(50))
layer = tf.nn.relu(tf.matmul(layer, w2) + b2)
# layer = tf.sigmoid(tf.matmul(layer, w2) + b2)
# layer = tf.nn.elu(tf.matmul(layer, w2) + b2)
# layer = tf.nn.selu(tf.matmul(layer, w2) + b2)


w = tf.Variable(tf.zeros([50,50]),name="weight")
b = tf.Variable(tf.zeros([50]),name="bias")
layer = tf.nn.softmax(tf.matmul(layer,w)+b)
# model.add(Dense(50))
layer = tf.nn.relu(tf.matmul(layer, w2) + b2)
# layer = tf.sigmoid(tf.matmul(layer, w2) + b2)
# layer = tf.nn.elu(tf.matmul(layer, w2) + b2)
# layer = tf.nn.selu(tf.matmul(layer, w2) + b2)


w = tf.Variable(tf.zeros([50,50]),name="weight")
b = tf.Variable(tf.zeros([50]),name="bias")
layer = tf.nn.softmax(tf.matmul(layer,w)+b)
# model.add(Dense(50))
layer = tf.nn.relu(tf.matmul(layer, w2) + b2)
# layer = tf.sigmoid(tf.matmul(layer, w2) + b2)
# layer = tf.nn.elu(tf.matmul(layer, w2) + b2)
# layer = tf.nn.selu(tf.matmul(layer, w2) + b2)


w = tf.Variable(tf.zeros([50,10]),name="weight")
b = tf.Variable(tf.zeros([10]),name="bias")
hypothesis = tf.nn.softmax(tf.matmul(layer,w)+b)
layer = tf.nn.relu(tf.matmul(layer, w2) + b2)
# layer = tf.sigmoid(tf.matmul(layer, w2) + b2)
# layer = tf.nn.elu(tf.matmul(layer, w2) + b2)
# layer = tf.nn.selu(tf.matmul(layer, w2) + b2)



loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(hypothesis),axis=1))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(loss)
# train = optimizer.minimize(cost)

# accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted,y),dtype=tf.float32))

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    for step in range(300):
        _,loss_val,hypo_val=sess.run([optimizer,loss,hypothesis],feed_dict={x:x_train,y:y_train})
        # if step % 10==1:
        #     print(loss_val)
        print(f"step:{step},loss_val:{loss_val}")
        # 실제로 실현되는 부분
    correct_prediction = tf.equal(tf.argmax(hypothesis,1),tf.argmax(y,1))
    
    #정확도
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print("Accuracy:",sess.run(accuracy,feed_dict={x:x_test,y:y_test}))

    # GYU code
    # predicted = tf.arg_max(hypo,1)
    # acc = tf.reduce_mean(tf.cast(tf.equal(predicted, tf.argmax(y,1)), dtype=tf.float32))


# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     predict = sess.run(hypothesis,feed_dict={x:x_test})
#     print(predict,sess.run(tf.argmax(predict,axis=1)))
