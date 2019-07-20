#!/usr/bin/python
# -*-coding:utf8-*-

"""
@author: LieOnMe
@time: 2019/7/20 22:32
"""

import os
import tensorflow as tf
from tensorflow import keras

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

(x, y), _ = keras.datasets.mnist.load_data(path="D:\\code\\workspace-self\\tensorflow-pg\\data\\mnist"
                                                "\\mnist.npz")

# convert to tensor
x = tf.convert_to_tensor(x, dtype=tf.float32) / 255
y = tf.convert_to_tensor(y, dtype=tf.float32)


train_db = tf.data.Dataset.from_tensor_slices((x, y)).batch(128)
train_iter = iter(train_db)
sample = next(train_iter)
print("Batch : ", sample[0].shape, sample[1].shape)

# [b, 784] -> [b, 256] -> [b, 128] -> [b, 10]
w1 = tf.Variable(tf.random.truncated_normal([784, 256], stddev=0.1))
b1 = tf.Variable(tf.zeros([256]))
w2 = tf.Variable(tf.random.truncated_normal([256, 128], stddev=0.1))
b2 = tf.Variable(tf.zeros([128]))
w3 = tf.Variable(tf.random.truncated_normal([128, 10], stddev=0.1))
b3 = tf.Variable(tf.zeros([10]))
lr = 1e-3

for (x, y) in train_db:
    # h1 = x@w1 + b1
    x = tf.reshape(x, [-1, 28*28])
    y_one_hot = tf.one_hot(y, depth=10)

    with tf.GradientTape() as tape: # 默认跟踪 tf.Variable
        h1 = x@w1 + b1
        h1 = tf.nn.relu(h1)
        h2 = h1@w2 + b2
        h2 = tf.nn.relu(h2)
        out = h2@w3 + b3

        # compute loss: mse
        loss = tf.square(y_one_hot - out)
        loss = tf.reduce_mean(loss)

    grads = tape.gradient(loss, [w1, b1, w2, b2, w3, b3])
    # w1 = w1 - lr * grads[0]
    w1.assign_sub(lr * grads[0])
    # b1 = b1 - lr * grads[1]
    # w2 = w2 - lr * grads[2]
    # b2 = b2 - lr * grads[3]
    # w3 = w3 - lr * grads[4]
    # b3 = b3 - lr * grads[5]
