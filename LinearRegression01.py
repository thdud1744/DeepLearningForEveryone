# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 19:51:58 2019

@author: soyeo
"""

import tensorflow as tf

x_train = [1, 2, 3]
y_train = [1, 2, 3]

# H(x) = Wx + b 
# W: weight b: bias
# random_normal([1]) : Rank 가 1인 랜덤한 수 생성. 값이 1개인 일차원 어레이
W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = x_train * W + b

# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - y_train))
# reduce_mean : 평균 내 주는 것.
# GradientDescentOptimizer: minimize 함수 찾는 것

# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost) # 얘도 노드이다!

# 여기까지가 graph 빌드

# Launch the graph in a session
sess = tf.Session()
# Initialize global variables in the graph 꼭 거쳐야 하는 과정.
sess.run(tf.global_variables_initializer())

# Fit the line
for step in range(2001):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(cost), sess.run(W), sess.run(b))
      
        
        
        
# VERSION 2        
# Placeholder 활용하기

W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')        
X = tf.placeholder(tf.float32, shape=[None])
Y = tf.placeholder(tf.float32, shape=[None])

hypothesis = X * W + b
cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost) # 얘도 노드이다!

sess = tf.Session()

sess.run(tf.global_variables_initializer())

for step in range(2001):
    cost_val, W_val, b_val, _ = sess.run([cost, W, b, train], feed_dict={X: [1, 2, 3, 4, 5], Y: [2.1, 3.1, 4.1, 5.1, 6.1]})
    if step % 20 == 0:
        print(step, cost_val, W_val, b_val)
        
# TEST!!!
        
print(sess.run(hypothesis, feed_dict={X:[5]}))
print(sess.run(hypothesis, feed_dict={X:[2.5]}))
print(sess.run(hypothesis, feed_dict={X:[1.5, 3.5]}))
