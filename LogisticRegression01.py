# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 21:08:05 2019

@author: soyeo
"""
# Lab 5, video 13
# Logistic Regression

import tensorflow as tf

# training data
x_data = [[1,2],[2,3],[3,1],[4,3], [5,3], [6,2]]
y_data = [[0],[0],[0],[1],[1],[1]] # bianry (class) 0 or 1, pass or fail

X = tf.placeholder(tf.float32, shape=[None,2])
Y = tf.placeholder(tf.float32, shape=[None,1])

W = tf.Variable(tf.random_normal([2,1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Hypothesis using sigmoid: tf.div(1., 1. + tf.exp(tf.matmul(X,W)+b))
hypothesis = tf.sigmoid(tf.matmul(X,W)+b)

# cost/loss function
cost = -tf.reduce_mean(Y*tf.log(hypothesis) + (1-Y)*tf.log(1-hypothesis))

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# Accuracy computation
# true if hypothesis > 0.5 else False
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
# 예측값 맞는지 확인하기. predicted와 Y값 같은지 확인하고, 그거 한번에 비율로 계산해줌
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

# launch graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(10001):
        cost_val, _ = sess.run([cost, train], feed_dict={X: x_data, Y:y_data})
        if step%200 == 0:
            print(step, cost_val)
            
# Accuarcy Report
h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X: x_data, Y:y_data})
print("\nHypothesis: ", h, "\nCorrect(Y): ", c, "\nAccuracy: ", a)


#------------------------------------------------------------------------------
import numpy as np
xy = np.loadtxt('data-03-diabetes.csv', delimiter=',', dtype=np.float32)
x_data = xy[:,0:-1]
y_data = xy[:, [-1]]

X = tf.placeholder(tf.float32, shape=[None, 8])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([8,1]), name='Weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.sigmoid(tf.matmul(X,W) +b)

cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1-Y) * tf.log(1-hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    feed = {X: x_data, Y: y_data}
    h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict = feed)
    for step in range(10001):
        sess.run(train, feed_dict = feed)
        if step%200 == 0:
            print(step, sess.run(cost, feed_dict = feed))
    print("\nHyptothesis: ",h, "\nCorrect(Y): ", c, "\nAccuracy: ", a)
