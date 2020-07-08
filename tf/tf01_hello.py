import tensorflow as tf
print(tf.__version__)
# 1.14.0

hello = tf.constant('Hello World') # 고정적인 , 값이 바뀌지 않음, 상수
print(hello)
# Tensor("Const:0", shape=(), dtype=string)
# 자료형이 나옴

sess = tf.Session()
print(sess.run(hello))
# b'Hello World'

# 
# input>sess>output이 tf 1.0대
# session 없앤게 케라스
# 케라스에선 tf가 백엔드로 돌아감