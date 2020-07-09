# 다중분류
# iris 코드 완성
from sklearn.datasets import load_iris
import tensorflow as tf

# 1. data
dataset = load_iris()
data = dataset.data
target = dataset.target

print(data.shape, target.shape) #(150, 4) (150,)

# one hot encoding

# aaa = tf.one_hot(y, ???)

x = tf.placeholder(tf.float32, shape=[None,4])
y = tf.placeholder(tf.float32, shape=[None,1])
w = tf.Variable(tf.random_normal([4,]), name ='weight')
b = tf.Variable(tf.random_normal([1,], name = 'bias'))

hypothesis = tf.nn.softmax(tf.matmul(x,w) + b)

cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(hypothesis), axis=1))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.3).minimize(cost)

train = optimizer.minimize(cost)


with tf.session() as sess:
    sess.run(tf.global_variables_initializer())

for step in range(2001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict = {x:data, y:target})
    if step % 200 == 0 :
        print(step, cost_val)

# R2, rmse, accuracy
a = sess.run(hypothesis, )
predicted = tf.arg_max(hypo,1)

acc = tf.reduce_mean(tf.cast(tf.equal(predicted, tf.argmax(y,1)), dtype=tf.float32))