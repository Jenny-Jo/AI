# (28,28,1)
# dnn layer 열개

from keras.datasets import mnist                         
mnist.load_data()                                         

(x_train, y_train), (x_test, y_test) = mnist.load_data()


print(x_train.shape)                   #(60000, 28, 28)
print(x_test.shape)                    #(10000, 28, 28)
print(y_train.shape)                   #(60000,) 스칼라, 1 dim(vector)
print(y_test.shape)                    #(10000,)

x_train = x_train.reshape(-1, x_train.shape[1]* x_train.shape[2])
x_test = x_test.reshape(-1, x_test.shape[1]* x_test.shape[2])


import tensorflow as tf
tf.set_random_seed(777)

x = tf.placeholder(tf.float32, shape=[28, 28])
y = tf.placeholder(tf.float32, shape=[None, 1])

w = tf.Variable(tf.zeros([28,1]),name = 'weight')
b = tf.Variable(tf.zeros([1]), name='bias')

w1 = tf.Variable(tf.zeros([28,100]),name = 'weight1', dtype=tf.float32)
b1 = tf.Variable(tf.zeros([100]), name='bias', dtype=tf.float32)
layer1 = tf.sigmoid(tf.matmul(x,w1) + b1)

w2 = tf.Variable(tf.zeros([100,100]), name ='weight2', dtype=tf.float32)
b2 = tf.Variable(tf.zeros([100], name = 'bias', dtype = tf.float32))
layer2 = tf.sigmoid(tf.matmul(layer1,w2) + b2)


w2 = tf.Variable(tf.zeros([100,100]), name ='weight2', dtype=tf.float32)
b2 = tf.Variable(tf.zeros([100], name = 'bias', dtype = tf.float32))
layer2 = tf.sigmoid(tf.matmul(layer2,w2) + b2)

w2 = tf.Variable(tf.zeros([100,100]), name ='weight2', dtype=tf.float32)
b2 = tf.Variable(tf.zeros([100], name = 'bias', dtype = tf.float32))
layer2 = tf.sigmoid(tf.matmul(layer2,w2) + b2)

w2 = tf.Variable(tf.zeros([100,100]), name ='weight2', dtype=tf.float32)
b2 = tf.Variable(tf.zeros([100], name = 'bias', dtype = tf.float32))
layer2 = tf.sigmoid(tf.matmul(layer2,w2) + b2)

w2 = tf.Variable(tf.zeros([100,100]), name ='weight2', dtype=tf.float32)
b2 = tf.Variable(tf.zeros([100], name = 'bias', dtype = tf.float32))
layer2 = tf.sigmoid(tf.matmul(layer2,w2) + b2)

w2 = tf.Variable(tf.zeros([100,100]), name ='weight2', dtype=tf.float32)
b2 = tf.Variable(tf.zeros([100], name = 'bias', dtype = tf.float32))
layer2 = tf.sigmoid(tf.matmul(layer2,w2) + b2)

w2 = tf.Variable(tf.zeros([100,100]), name ='weight2', dtype=tf.float32)
b2 = tf.Variable(tf.zeros([100], name = 'bias', dtype = tf.float32))
layer2 = tf.sigmoid(tf.matmul(layer2,w2) + b2)


w2 = tf.Variable(tf.zeros([100,100]), name ='weight2', dtype=tf.float32)
b2 = tf.Variable(tf.zeros([100], name = 'bias', dtype = tf.float32))
layer2 = tf.sigmoid(tf.matmul(layer2,w2) + b2)

w2 = tf.Variable(tf.zeros([100,1]), name ='weight2', dtype=tf.float32)
b2 = tf.Variable(tf.zeros([1], name = 'bias', dtype = tf.float32))
layer2 = tf.sigmoid(tf.matmul(layer2,w2) + b2)


hypothesis = tf.sigmoid(tf.matmul(x,w) + b)

# cost = tf.reduce_mean(tf.square(hypothesis-y))
cost = -tf.reduce_mean(y*tf.log(hypothesis)+
                       (1-y)*tf.log(1-hypothesis))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(cost)

predicted = tf.cast(hypothesis >=0.5, dtype = tf.float32)

accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted,y), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(10000):
        cost_val, _ , acc = sess.run([cost,train,accuracy], feed_dict={x:x_train, y:y_train})

        if step % 200 == 0 :
            print(step, cost_val, acc)
    h, c, a = sess.run([hypothesis,predicted, accuracy], feed_dict={x:x_train, y:y_train})

print('\n hypothesis: ', h, '\n correct(y): ',c, '\n accuracy: ',a)