# 이진분류
# 과제
from sklearn.datasets import load_breast_cancer
import tensorflow as tf

dataset = load_breast_cancer()
data = dataset.data
target = dataset.target

target = target.reshape(-1, 1)

print(data.shape, target.shape) # (569, 30) (569, 1)

x = tf.placeholder(tf.float32, shape=[None,30])
y = tf.placeholder(tf.float32, shape=[None])


w = tf.Variable(tf.random_normal([30,1]), name ='weight') # x data의 3열을 여기선 3행으로?
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.sigmoid(tf.matmul(x,w) + b)
# hypothesis = w*x + b

cost = tf.reduce_mean(tf.square(hypothesis-y))

cost = -tf.reduce_mean(y*tf.log(hypothesis)+
                       (1-y)* tf.log(1-hypothesis)) # crossentropy 수식 # - 붙은 이유 : 음수 안나오게
                                                        # 0.00001 # sigmoid거쳐 나온 애 무조건 0~1 사이라 log에 들어가면 음수나옴
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-6)
train = optimizer.minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype = tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted,y), dtype=tf.float32))

with tf.Session() as sess:  
    sess.run(tf.global_variables_initializer())
 
    for step in range(5001):
        cost_val,_  = sess.run([cost,train], feed_dict={x: data, y:target})
        if step % 200 ==0:
            print(step, 'cost :', cost_val)
    h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict = {x:data, y:target})
    print('\n Hypothesis:', h, '\n Correct(y):',c, '\n Accuracy: ', a)