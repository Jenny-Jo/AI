import tensorflow as tf
node1 = tf.constant(3.0, tf.float32) # defalut = float32
node2 = tf.constant(3.0)             # 두 tensor의 type이 같아야 한다
node3 = tf.add(node1, node2)

print('node1:', node1, 'node2', node2)
print('node3:',node3)
# node1: Tensor("Const:0", shape=(), dtype=float32) node2 Tensor("Const_1:0", shape=(), dtype=float32)
# node3: Tensor("Add:0", shape=(), dtype=float32)

#input만 해줌
sess = tf.Session()
print('sess.run(node1, node2):', sess.run([node1, node2]))
print('sess.run(node3)', sess.run(node3))
# sess.run(node1, node2): [3.0, 3.0]
# sess.run(node3) 6.0