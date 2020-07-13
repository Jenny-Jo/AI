# RNN 모델로 짜기
# 숙제

import tensorflow as tf
import numpy as np

dataset = np.array([1,2,3,4,5,6,7,8,9,10])

dataset = np.array([1,2,3,4,5,6,7,8,9,10])

def split_xy1(dataset, time_steps):
    x, y = list(), list()
    for i in range(len(dataset)):
        end_number = i + time_steps
        if end_number > len(dataset) -1 :
            break
        tmp_x, tmp_y = dataset [i:end_number], dataset[end_number]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)
x,y = split_xy1(dataset, 5)
print(x, '\n', y)
'''
[[1 2 3 4 5]
 [2 3 4 5 6]
 [3 4 5 6 7]
 [4 5 6 7 8]
 [5 6 7 8 9]] 
 [ 6  7  8  9 10]
'''

x =  x.reshape(5,5,1)
y = y.reshape(5,1)
print( x.shape)     # (5, 5)
print(y.shape)     #  (5, 1)

sequence_length = 5
input_dim = 1
output = 1
batch_size = 5 # 전체 행

X = tf.compat.v1.placeholder(tf.float32,(None,sequence_length,input_dim))
Y = tf.compat.v1.placeholder(tf.int32,(None, 1))
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
'''
sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=hypothesis, targets=Y, weights= weights) 
# loss=0, 모든 예측값(hypothesis) 다 맞아떨어지면 acc=1, mse=0, 결과가 너무 잘 나오면 과적합이다
cost = tf.reduce_mean(sequence_loss)
'''
cost = tf.reduce_mean(tf.square(hypothesis - y))

train = tf.compat.v1.train.AdamOptimizer(learning_rate=0.1).minimize(cost)

# prediction = tf.argmax(hypothesis, axis=1) # 3차원 (1,6,5)라서 axis =2

# 3-2. Train
with tf.Session() as sess:
    # loss, 
    sess.run(tf.global_variables_initializer())
    for i in range(401):
        loss, _ = sess.run([cost, train], feed_dict={X: x, Y: y}) # graph 연산 - tensor 연산
        # result = sess.run(prediction, feed_dict={X: x})
        print(i, 'loss: ', loss,  'ture Y : ',  y.reshape(-1))#, 'prediction:', result.reshape(-1))

        # result_str = [idx2char[c] for c in np.squeeze(result)]
        # print('\nPrediction str: ', ''.join(result_str))

# 그림 그려보기

