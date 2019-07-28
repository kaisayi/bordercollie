#!/usr/bin/python
# -*-coding:utf8-*-

"""
@author: LieOnMe
@time: 2019/7/29 1:54
"""

import os

import tensorflow as tf
from tensorflow import keras

from utils import conf
from model.residual_net import resnet18

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
PROJ_PATH = conf.get_project_path()
tf.random.set_seed(2345)


def pre_process(x, y):
    x = 2 * tf.cast(x, dtype=tf.float32) / 255. - 1
    y = tf.cast(y, dtype=tf.int32)
    return x, y


(x, y), (x_test, y_test) = keras.datasets.cifar100.load_data()
y = tf.squeeze(y, axis=1)
y_test = tf.squeeze(y_test, axis=1)
print(x.shape, y.shape, x_test.shape, y_test.shape)

train_db = tf.data.Dataset.from_tensor_slices((x, y))
train_db = train_db.shuffle(1000).map(pre_process).batch(64)
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_db = test_db.map(pre_process).batch(64)


def main():
    model = resnet18()
    model.build(input_shape=(None, 32, 32, 3))
    model.summary()
    optimizer = keras.optimizers.Adam(lr=1e-4)

    for epoch in range(50):
        for step, (_x, _y) in enumerate(train_db):
            with tf.GradientTape() as tape:
                # [b, 512] => [b, 100]
                logits = model(_x)
                y_onehot = tf.one_hot(_y, depth=100)
                # compute loss
                loss = tf.losses.categorical_crossentropy(y_onehot, logits, from_logits=True)
                loss = tf.reduce_mean(loss)

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if step % 100 == 0:
                print("Epoch: {}, Step: {}, loss: {}"
                      .format(epoch, step, float(loss)))

        total_correct = 0
        total_num = 0
        for xt, yt in test_db:
            logits = model(xt)
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
