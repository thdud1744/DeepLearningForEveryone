# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 16:11:04 2019

@author: soyeo
"""

# fancy softmax classifier
# 17th video

import tensorflow as tf
import numpy as np


xy = np.loadtxt('data-04-zoo.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]
nb_classes = 7

X = tf.placeholder(tf.float32, [None, 16])
Y = tf.placeholder(tf.int32, [None, 1]) # possible : 0 ~ 6, shape=(?,1)

Y_one_hot = tf.one_hot(Y, nb_classes) # one hot shape = (?, 1, 7) one_hot 변환하며 한 차원 늘어난다.
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes]) # shape = (?,7) 다시 차원 축소

W = tf.Variable(tf.random_normal([16, nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

logits = tf.matmul(X, W) + b
hypothesis = tf.nn.softmax(logits)

# 1. Cross entropy cost / loss ------------------------------------------------
cost = tf.reduce_mean( -tf.reduce_sum( Y * tf.log(hypothesis), axis=1))
# Instead, use softmax_cross_entropy_with_logits() ------------------------------
cost_i = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y_one_hot)
cost = tf.reduce_mean(cost_i)
#-------------------------------------------------------------------------------

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

prediction = tf.argmax(hypothesis, 1)
correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    feed = {X: x_data, Y:y_data}
    for step in range(2001):
        sess.run(optimizer, feed_dict=feed)
        if step%100 == 0:
            loss, acc = sess.run([cost, accuracy], feed_dict=feed)
            print("Step: {:5}\tLoss: {:.3f}\tAcc: {:.2%}".format(step, loss, acc))
    pred = sess.run(prediction, feed_dict = {X:x_data})
    for p, y in zip(pred, y_data.flatten()):
        print("[{}] Prediction: {} True Y:{}".format(p== int(y), p , int(y)))
            
