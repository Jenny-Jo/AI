import tensorflow as tf
import numpy as np

# data = hihello

idx2char = ['e', 'h','i','l','o']

# _data = np.array([['h'], ['i'], ['h'], ['e'], ['l'], ['l'], ['o']])

_data = np.array([['h','i','h','e','l','l','o']], dtype=np.str).reshape(-1,1)

print(_data.shape)      # (1, 7)
print(_data)            # [['h' 'i' 'h' 'e' 'l' 'l' 'o']]
print(type(_data))      # <class 'numpy.ndarray'>

from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
enc.fit(_data)
_data = enc.transform(_data).toarray()
print('-'*50)
print(_data)            # [[1. 1. 1. 1. 1. 1. 1.]]
print(type(_data))      # <class 'numpy.ndarray'>
print(_data.dtype)      # float64
'''
(7, 1)
[['h']
 ['i']
 ['h']
 ['e']
 ['l']
 ['l']
 ['o']]
<class 'numpy.ndarray'>
--------------------------------------------------
[[0. 1. 0. 0. 0.]
 [0. 0. 1. 0. 0.]
 [0. 1. 0. 0. 0.]
 [1. 0. 0. 0. 0.]
 [0. 0. 0. 1. 0.]
 [0. 0. 0. 1. 0.]
 [0. 0. 0. 0. 1.]]
<class 'numpy.ndarray'>
float64'''

x_data = _data[:6, ] # hihell
y_data = _data[1:, ] #  ihello

print('---x---')
print(x_data)
print('---y---')
print(y_data)

y_data = np.argmax(y_data, axis=1)
print('-'*50,'y argmax','-'*50)
print(y_data)            # [2 1 0 3 3 4]
print(y_data.shape)      # (6, )

x_data = x_data.reshape(1,6,5)
y_data = y_data.reshape(1,6)

print(x_data) 
                        # [[[0. 1. 0. 0. 0.]
                        #   [0. 0. 1. 0. 0.]
                        #   [0. 1. 0. 0. 0.]
                        #   [1. 0. 0. 0. 0.]
                        #   [0. 0. 0. 1. 0.]
                        #   [0. 0. 0. 1. 0.]]]
print(x_data.shape)     # (1, 6, 5)
print(y_data.shape)     # (1, 6)

sequence_length = 6
input_dim = 5
output = 5
batch_size = 1 # 전체 행

X = tf.compat.v1.placeholder(tf.float32,(None,sequence_length,input_dim))
Y = tf.compat.v1.placeholder(tf.int64,(None, sequence_length))
                                # argmax 때문에??
print(X)
print(Y)
# Tensor("Placeholder:0", shape=(?, 6, 5), dtype=float32)
# Tensor("Placeholder_1:0", shape=(?, 6), dtype=float32)

# 2. MODEL
# model.add(LSTM(outut,                     input_shape=(6,5)))
# cell = tf.nn.rnn_cell.BasicLSTMCell(output) # 중간과정
cell = tf.keras.layers.LSTMCell(output)
hypothesis, _states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
                            # model.add(LSTM)
print(hypothesis)
# Tensor("rnn/transpose_1:0", shape=(?, 6, 5), dtype=float32) output이 5

# 3-1. Compile
weights = tf.ones([batch_size, sequence_length]) 

sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=hypothesis, targets=Y, weights= weights) 
# loss=0, 모든 예측값(hypothesis) 다 맞아떨어지면 acc=1, mse=0, 결과가 너무 잘 나오면 과적합이다
cost = tf.reduce_mean(sequence_loss)

train = tf.compat.v1.train.AdamOptimizer(learning_rate=0.1).minimize(cost)

prediction = tf.argmax(hypothesis, axis=2) # 3차원 (1,6,5)라서 axis =2

# 3-2. Train
with tf.Session() as sess:
    # loss, ?????
    sess.run(tf.global_variables_initializer())
    for i in range(401):
        loss, _ = sess.run([cost, train], feed_dict={X:x_data, Y:y_data}) # graph 연산 - tensor 연산
        result = sess.run(prediction, feed_dict={X:x_data})
        print(i, 'loss: ', loss, 'prediction:', result, 'ture Y : ', y_data)

        result_str = [idx2char[c] for c in np.squeeze(result)]
        print('\nPrediction str: ', ''.join(result_str))

# 그림 그려보기 

