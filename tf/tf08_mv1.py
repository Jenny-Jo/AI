# multi variable
import tensorflow as tf
tf.set_random_seed(777)

x1_data = [73., 93., 89., 96., 73.]
x2_data = [80., 88., 91., 98., 66.]
x3_data = [75., 93., 90., 100., 70.]

y_data = [152., 185., 180., 196., 142.]

x1 = tf.placeholder(tf.float32)
x2 = tf.placeholder(tf.float32)
x3 = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

w1 = tf.Variable(tf.random_normal([1]), name='weight1')
w2 = tf.Variable(tf.random_normal([1]), name='weight1')
w3 = tf.Variable(tf.random_normal([1]), name='weight1')
b = tf.Variable(tf.random_normal([1]), name='weight1')

hypothesis = x1*w1 + x2*w2 + x3*w3 + b 

cost = tf.reduce_mean(tf.square(hypothesis-y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

with tf.Session() as sess:                        # with을 쓰면 open, close를 안써도 됌 / Session을 계속 사용하기 위해 열어둔다
    sess.run(tf.global_variables_initializer()) 
    
    for step in range(2001):
        cost_val, hy_val,_  = sess.run([cost, hypothesis, train],
                                    feed_dict={x1: x1_data, x2:x2_data, x3:x3_data, y:y_data})
        if step % 10 ==0:
            print(step, 'cost :', cost_val, '\n ', 'hy_val:', hy_val)
# 큰 데이터할 땐  with 넣어줘야 해