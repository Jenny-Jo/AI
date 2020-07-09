# multi variable
import tensorflow as tf
import numpy as np
tf.set_random_seed(777)

dataset = np.loadtxt('F:/Study/data/csv/data-01-test-score.csv',
                     delimiter=',', dtype=np.float32)
x_data = dataset[:, 0:-1]
y_data = dataset[:,[-1]]

#######################################################################################

x = tf.placeholder(tf.float32, shape=[None,3])
y = tf.placeholder(tf.float32, shape=[None,1])

w = tf.Variable(tf.random_normal([3,1]), name ='weight') # x data의 3열을 여기선 3행으로?
b = tf.Variable(tf.random_normal([1]), name='bias')

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
                                feed_dict={x: x_data, y:y_data})
    if step % 10 ==0:
        print(step, 'cost :', cost_val, '\n 예측값 ', 'hy_val:', hy_val)

# activation linear로 하고 있다/ 계단함수에서 sigmoid로 발전> 중간값 없어져서 relu> 음수 못해서 leaky relu, selu 나오고...
# 레이어 하나로 하고 있음
# 


