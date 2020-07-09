import tensorflow as tf
tf.set_random_seed(777)

                                                    # 우리가 사용하는 변수와 동일
W = tf.Variable(tf.random_normal([1]), name = 'weight') # 단, Variable사용시 초기화 필수
b = tf.Variable(tf.random_normal([1]), name = 'bias')
                        #_normalization

print(W)
# <tf.Variable 'weight:0' shape=(1,) dtype=float32_ref>
# 자료형만 나옴

# Session & .run
W = tf.Variable([0.3], tf.float32)
sess = tf.Session() # 메모리 열어주고,
sess.run(tf.global_variables_initializer()) # 변수 초기화
aaa = sess.run(W)
print('aaa',aaa) # [0.3]
sess.close() # 메모리 닫아줌 => with로 대체 가능

# set로 묶임 inter~ & .eval
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
bbb = W.eval()
print('bbb',bbb) # [0.3]
sess.close()
# WARNING:tensorflow:From f:\Study\tf\tf06_Variable.py:22: The name tf.InteractiveSession is deprecated. Please use tf.compat.v1.InteractiveSession instead.
# warning 무시해라

sess = tf.Session()
sess.run(tf.global_variables_initializer())
ccc = W.eval(session = sess)
print('ccc',ccc) # [0.3]
sess.close()
