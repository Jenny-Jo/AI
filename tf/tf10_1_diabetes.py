# 분류
# 과제
from sklearn.datasets import load_diabetes
import tensorflow as tf

dataset = load_diabetes()
data = dataset.data
target = dataset.target

target = target.reshape(-1, 1)

print(data.shape, target.shape) # (442, 10) (442, 1)

tf.set_random_seed(777)

x = tf.placeholder(tf.float32, shape=[None, 10])
y = tf.placeholder(tf.float32, shape=[None, 1])

w = tf.Variable(tf.random_normal([10,1]),name = 'weight')
b = tf.Variable(tf.random_normal([1]),name ='bias')

hypothesis = tf.matmul(x,w) + b
# hypothesis = w*x + b

cost = tf.reduce_mean(tf.square(hypothesis-y))
                                                        # 0.00001
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
 
for step in range(2001):
    cost_val, hy_val,_  = sess.run([cost, hypothesis, train],
                                feed_dict={x: data, y:target})
    if step % 10 ==0:
        print(step, 'cost :', cost_val, '\n 예측값 ', 'hy_val:', hy_val)