# lr 수정해서 연습
# 0.01 > 0.1/ 0.01/1
# epoch 2000보다 적게 만들기
import tensorflow as tf
tf.set_random_seed(777)

# x_train = [1, 2, 3]
# y_train = [3, 5, 7]

x_train = tf.placeholder(tf.float32, shape =[None]) # input과 비슷한 개념
y_train = tf.placeholder(tf.float32, shape =[None])

                                                        # 우리가 사용하는 변수와 동일
W = tf.Variable(tf.random_normal([1]), name = 'weight') # 단, Variable사용시 초기화 필수
b = tf.Variable(tf.random_normal([1]), name = 'bias')
                        #_normalization

# sess = tf.Session()
# sess.run(tf.global_variables_initializer()) # 변수 초기화
# print(sess.run(W))                          # [2.2086694]

hypothesis = x_train * W + b                  # model

cost = tf.reduce_mean(tf.square(hypothesis - y_train))   # cost = loss
                                                         # mse

train = tf.train.GradientDescentOptimizer(learning_rate= 0.1).minimize(cost) # cost값 최소화
        # cost를 최소화하기 위해 각 Variable을 천천히 변경하는 optimizer 
        # 참고 https://www.youtube.com/watch?v=TxIVr-nk1so&list=PLlMkM4tgfjnLSOjrEJN31gZATbcj_MpUm&index=6

with tf.Session() as sess:                        # with을 쓰면 open, close를 안써도 됌 / Session을 계속 사용하기 위해 열어둔다
# with tf.combat.v1.Session() as sess:                 # v1 쓸 수도 있다 

    sess.run(tf.global_variables_initializer())   # 변수 선언 # 이 이후로 모든 변수들 초기화 
    # sess.run(tf.combat.v1.global_variables_initializer())   # 변수 선언 # 이 이후로 모든 변수들 초기화 
                                     
    for step in range(1001):
        _, cost_val, W_val, b_val = sess.run([train, cost, W, b], feed_dict={x_train:[1,2,3], y_train:[3,5,7]}) # session을 이용해 train 훈련 
                                                                            # placeholder로 넣어줘서 feeddict로 넣어줘야 함
        if step % 20 == 0:
            print(step, cost_val, W_val, b_val)
            
# 행렬 연산 - 넘파이식 곱셈과 일반 행렬 곱셈 다름 ( 참조 : https://chan-lab.tistory.com/8 )

# predict 해보자
    print("predict: ",sess.run(hypothesis, feed_dict= {x_train:[4]})) # 9
    #predict:  [9.000078]
    print("predict: ",sess.run(hypothesis, feed_dict= {x_train:[5,6]}))#11,13
    print("predict: ",sess.run(hypothesis, feed_dict= {x_train:[6,7,8]}))#13,15,17
#     predict:  [11.000123 13.000169]
#     predict:  [13.000169 15.000214 17.000257]
    
