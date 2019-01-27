# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 19:40:12 2019

@author: soyeo
"""
import tensorflow as tf
# Queue Runner

# 파일이 굉장히 크거나, 여러 개 있을 때. 메모리에 한번에 올리기 힘들때
# 여러 개 파일을 큐에 쌓아두고, Reader로 연결해 디코딩 한 후 batch 만큼 큐에서 읽어옴.

# STEP 1: make list of files
filename_queue = tf.train.string_input_producer(['data-01-test-score.csv'], shuffle=False, name='filename_queue')

# STEP 2: define Readers to read files
reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

# STEP 3: parse the value
record_defaults = [[0.], [0.], [0.], [0.]]
xy = tf.decode_csv(value, record_defaults=record_defaults)


# collect batches of csv
train_x_batch, train_y_batch = tf.train.batch([xy[0:-1], xy[-1:]], batch_size=10)

# placeholders for tensor
X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([3,1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')
 
hypothesis = tf.matmul(X,W) + b

#------------------------------------------------------------------------------
# cost/loss funciton
cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
# start populating the filename queue
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)
# ERROR : Expect 2 fields but have 4 in record 0
 
for step in range(2001):
    x_batch, y_batch = sess.run([train_x_batch, train_y_batch])
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict={X: x_batch, Y:y_batch})
    if step%10 ==0:
        print(step, "Cost: ", cost_val, "\nPrediction:\n", hy_val)
    
coord.request_stop()
coord.join(threads)


# Ask my score
print("Your score will be ",
      sess.run(hypothesis, feed_dict={X: [[100, 70, 101]]}))

print("Other scores will be ",
      sess.run(hypothesis, feed_dict={X: [[60, 70, 110], [90, 100, 80]]}))
