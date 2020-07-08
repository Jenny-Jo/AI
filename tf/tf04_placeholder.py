import tensorflow as tf
node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(3.0)
node3 = tf.add(node1, node2)

sess = tf.Session()

a = tf.placeholder(tf.float32) # input과 비슷한 개념
b = tf.placeholder(tf.float32) # a, b 는 변수가 아님 placeholder라는 형식

adder_node = a + b
print(sess.run(adder_node, feed_dict={a:3, b:4.5})) # feed_dict = input과 비슷
print(sess.run(adder_node, feed_dict={a:[1,3], b:[2,4]})) # 리스트도 들어갈 수 있어

add_and_triple = adder_node * 3 # (a + b)*3
print(sess.run(add_and_triple, feed_dict={a:3, b:4.5}))

'''
넘파이식 계산
7.5
[3. 7.]
22.5
'''
'''
엑스와 와이를 넣고 웨이트 그라디언트 나와...?
와이는 더블유엑스 더하기비 디엔엔 베이스
컴파일 엠에스이...왜 인지 몰랐다...그거를 텐서폴로로 해보자
'''
