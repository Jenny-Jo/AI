
import tensorflow as tf

tf.set_random_seed(777)

x_train = [1, 2, 3]
y_train = [3, 5, 7]



# w = tf.Variable(tf.random_normal([3]), name='weight')
# b = tf.Variable(tf.random_normal([3]), name='bias')

w = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# 초기화? 뭐라는거야 
sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# print(sess.run(w))

hypothesis = w * x_train + b # 이게 모델이다...

cost = tf.reduce_mean(tf.square(hypothesis - y_train)) # mse

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost) # 경사하강법 인데 최소의 cost를 찾겠다 # 여기가 제일 중요!
# train = tf.train.MomentumOptimizer(learning_rate=0.01).minimize(cost)

with tf.Session() as sess: # 전체가 범위안에 포함된다? 이해안됨 안에있는 세션이 다 실행한다 ??
    sess.run(tf.global_variables_initializer()) # 변수들을 초기화 

    for step in range(2001):
        _, cost_val, w_val, b_val = sess.run([train, cost, w, b])

        if step % 20 == 0:
            print(step, cost_val, w_val, b_val)