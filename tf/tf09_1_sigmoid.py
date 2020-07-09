# multi variable
import tensorflow as tf
tf.set_random_seed(777)

x_data = [[1,2],
          [2,3],
          [3,1],
          [4,3],
          [5,3],
          [6,2]
          ]
y_data = [[0],
          [0],
          [0],
          [1],
          [1],
          [1]]

x = tf.placeholder(tf.float32, shape=[None,2])
y = tf.placeholder(tf.float32, shape=[None,1])

w = tf.Variable(tf.random_normal([2,1]), name ='weight') # x data의 3열을 여기선 3행으로?
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.sigmoid(tf.matmul(x,w) + b)
# hypothesis = w*x + b

cost = tf.reduce_mean(tf.square(hypothesis-y))

cost = -tf.reduce_mean(y*tf.log(hypothesis)+
                       (1-y)* tf.log(1-hypothesis)) # crossentropy 수식 # - 붙은 이유 : 음수 안나오게
                                                        # 0.00001 # sigmoid거쳐 나온 애 무조건 0~1 사이라 log에 들어가면 음수나옴
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.05)
train = optimizer.minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype = tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted,y), dtype=tf.float32))

with tf.Session() as sess:  
    sess.run(tf.global_variables_initializer())
 
    for step in range(5001):
        cost_val,_  = sess.run([cost,train], feed_dict={x: x_data, y:y_data})
        if step % 200 ==0:
            print(step, 'cost :', cost_val)
    h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict = {x:x_data, y:y_data})
    print('\n Hypothesis:', h, '\n Correct(y):',c, '\n Accuracy: ', a)

'''
 Hypothesis: [[0.9206136 ]
 [0.97887874]
 [0.96542114]
 [0.99525297]
 [0.99776244]
 [0.9980228 ]]
 Correct(y): [[1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]]
 Accuracy:  0.5
'''
