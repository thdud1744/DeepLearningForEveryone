# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 18:57:33 2019

@author: soyeo
"""

import tensorflow as tf
tf.__version__

hello = tf.constant("Hello, TensorFlow!")

sess = tf.Session()
print(sess.run(hello))

# Build Graph

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0) # tf.float32 implicitly
node3 = tf.add(node1, node2) # node3 = node1 + node2

print("node1: ", node1, "node2: ", node2)
print("node3: ", node3)

# Run Graph (sess.run())

sess = tf.Session()
print("sess.run(node1, node2): ", sess.run([node1, node2]))
print("sess.run(node3): ", sess.run(node3))

# Update Graph


# Placeholder 미리 값을 넣어두는게 아니라 실행시키는 단계에서 값을 넣어주고싶다. 
# 'node'를 만드는데 'placeholder'라는 특별한 노드로 만들어준다.
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b

print(sess.run(adder_node, feed_dict={a: 3, b:4.5}))
print(sess.run(adder_node, feed_dict={a: [1, 3], b: [2, 4]}))

# Everything in Tensor
# Rank, Shapes, Types

# Scalar: 0 magnitude []
# Vector: 1 magnitude + direction [D0]
# Matrix: 2 table of numbers [D0, D1]
# 3-Tensor: 3 cube of numbers [D0, D1, D2]
# n-Tensor: n [D0, D1, D2, ..., Dn-1]

