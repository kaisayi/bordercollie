#!/usr/bin/python
# -*-coding:utf8-*-

"""
@author: LieOnMe
@time: 2019/7/28 23:04
"""

import os
import tensorflow as tf
from tensorflow import keras
from utils import conf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
PROJ_PATH = conf.get_project_path()
tf.random.set_seed(2345)

conv_layers = [
    # Unit 1
    keras.layers.Conv2D(64, kernel_size=[3, 3], padding='same', activation=tf.nn.relu),
    keras.layers.Conv2D(64, kernel_size=[3, 3], padding='same', activation=tf.nn.relu),
    keras.layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

    # Unit 2
    keras.layers.Conv2D(128, kernel_size=[3, 3], padding='same', activation=tf.nn.relu),
    keras.layers.Conv2D(128, kernel_size=[3, 3], padding='same', activation=tf.nn.relu),
    keras.layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

    # Unit 3
    keras.layers.Conv2D(256, kernel_size=[3, 3], padding='same', activation=tf.nn.relu),
    keras.layers.Conv2D(256, kernel_size=[3, 3], padding='same', activation=tf.nn.relu),
    keras.layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

    # Unit 4
    keras.layers.Conv2D(512, kernel_size=[3, 3], padding='same', activation=tf.nn.relu),
    keras.layers.Conv2D(512, kernel_size=[3, 3], padding='same', activation=tf.nn.relu),
    keras.layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

    # Unit 5
    keras.layers.Conv2D(512, kernel_size=[3, 3], padding='same', activation=tf.nn.relu),
    keras.layers.Conv2D(512, kernel_size=[3, 3], padding='same', activation=tf.nn.relu),
    keras.layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same')
]


def pre_process(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255.
    y = tf.cast(y, dtype=tf.int32)
    return x, y


(x, y), (x_test, y_test) = keras.datasets.cifar100.load_data()
y = tf.squeeze(y, axis=1)
y_test = tf.squeeze(y_test, axis=1)
print(x.shape, y.shape, x_test.shape, y_test.shape)

train_db = tf.data.Dataset.from_tensor_slices((x, y))
train_db = train_db.shuffle(1000).map(pre_process).batch(128)
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_db = test_db.map(pre_process).batch(128)


def main():
    conv_nets = keras.Sequential(conv_layers)

    fc_nets = keras.Sequential([
        keras.layers.Dense(256, activation=tf.nn.relu),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(100, activation=tf.nn.relu)
    ])

    conv_nets.build(input_shape=[None, 32, 32, 3])
    fc_nets.build(input_shape=[None, 512])
    optimizer = keras.optimizers.Adam(lr=1e-4)

    variables = conv_nets.trainable_variables + fc_nets.trainable_variables

    for epoch in range(50):
        for step, (_x, _y) in enumerate(train_db):
            with tf.GradientTape() as tape:
                out = conv_nets(_x)
                # flatten
                out = tf.reshape(out, [-1, 512])
                # [b, 512] => [b, 100]
                logits = fc_nets(out)
                y_onehot = tf.one_hot(_y, depth=100)
                # compute loss
                loss = tf.losses.categorical_crossentropy(y_onehot, logits, from_logits=True)
                loss = tf.reduce_mean(loss)

            grads = tape.gradient(loss, variables)
            optimizer.apply_gradients(zip(grads, variables))

            if step % 100 == 0:
                print("Epoch: {}, Step: {}, loss: {}"
                      .format(epoch, step, float(loss)))

        total_correct = 0
        total_num = 0
        for xt, yt in test_db:
            out = conv_nets(xt)
            out = tf.reshape(out, [-1, 512])
            logits = fc_nets(out)
            prob = tf.nn.softmax(logits, axis=1)
            pred = tf.argmax(prob, axis=1)
            pred = tf.cast(pred, dtype=tf.int32)

            correct = tf.cast(tf.equal(pred, yt), dtype=tf.int32)
            correct = tf.reduce_sum(correct)

            total_num += xt.shape[0]
            total_correct += int(correct)

        acc = total_correct / total_num
        print("Epoch: {}, acc: {}"
              .format(epoch, acc))


if __name__ == "__main__":
    main()
