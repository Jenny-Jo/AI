# hypothesis를 구하시오
# H = Wx + b
# aaa, bbb, ccc 자리에 각 hypothesis를 구하시오
import tensorflow as tf
tf.set_random_seed(777)

x = [1,2,3]
W = tf.Variable([0.3], tf.float32)
b = tf.Variable([1], tf.float32)



# Session & .run
sess = tf.Session() # 메모리 열어주고,
sess.run(tf.global_variables_initializer()) # 변수 초기화
hypothesis = sess.run(W)
print('hypothesis: ',hypothesis) # [0.3]
sess.close() # 메모리 닫아줌 => with로 대체 가능

# set로 묶임 inter~ & .eval
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
hypothesis = W.eval()
print('hypothesis: ',hypothesis) # [0.3]
sess.close()


sess = tf.Session()
sess.run(tf.global_variables_initializer())
hypothesis = W.eval(session = sess)
print('hypothesis:',hypothesis) # [0.3]
sess.close()
