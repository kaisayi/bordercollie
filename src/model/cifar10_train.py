#!/usr/bin/python
# -*-coding:utf8-*-

"""
@author: LieOnMe
@time: 2019/7/27 17:07
"""

import os
import tensorflow as tf
from tensorflow import keras
from utils import conf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
PROJ_PATH = conf.get_project_path()


def pre_process(x, y):
    # [0-255] => [-1, 1]
    x = 2 * tf.cast(x, dtype=tf.float32) / 255 - 1.
    y = tf.cast(y, dtype=tf.int32)
    y = tf.squeeze(y)
    y = tf.one_hot(y, depth=10)
    return x, y


data_path = os.path.normpath(os.path.join(PROJ_PATH, 'data/cifar/cifar-10-python.tar.gz'))
print(data_path)
(x, y), (x_test, y_test) = keras.datasets.cifar10.load_data()
print(x.shape, y.shape)
print(x_test.shape, y_test.shape)

batch_size = 64
db = tf.data.Dataset.from_tensor_slices((x, y))
db = db.map(pre_process).shuffle(10000).batch(batch_size)

db_val = tf.data.Dataset.from_tensor_slices((x_test, y_test))
db_val = db_val.map(pre_process).batch(batch_size)

db_iter = iter(db)
sample = next(db_iter)
print("Sample shape: ", sample[0].shape, sample[1].shape)


class MyDense(keras.layers.Layer):

    def __init__(self, inp_dim, outp_dim):
        super(MyDense, self).__init__()

        self.kernel = self.add_variable('w', [inp_dim, outp_dim])
        # self.bias = self.add_variable('b', [outp_dim])

    def call(self, inputs, **kwargs):
        out = inputs @ self.kernel
        return out

    def compute_output_signature(self, input_signature):
        pass


class MyNetwork(keras.Model):

    def __init__(self):
        super(MyNetwork, self).__init__()
        self.fc1 = MyDense(32 * 32 * 3, 256)
        self.fc2 = MyDense(256, 128)
        self.fc3 = MyDense(128, 64)
        self.fc4 = MyDense(64, 32)
        self.fc5 = MyDense(32, 10)

    def call(self, inputs, training=None, mask=None):
        """
        :param inputs: [b, 32, 32, 3]
        :param training:
        :param mask:
        :return:
        """
        x = tf.reshape(inputs, [-1, 32 * 32 * 3])
        x = self.fc1(x)
        x = tf.nn.relu(x)
        x = self.fc2(x)
        x = tf.nn.relu(x)
        x = self.fc3(x)
        x = tf.nn.relu(x)
        x = self.fc4(x)
        x = tf.nn.relu(x)
        x = self.fc5(x)
        return x

    def compute_output_signature(self, input_signature):
        pass


network = MyNetwork()
network.compile(keras.optimizers.Adam(lr=1e-3),
                loss=tf.losses.CategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

network.fit(db, epochs=15, validation_data=db_val, validation_freq=1)

weight_store_path = os.path.normpath(os.path.join(PROJ_PATH, 'out/cifar10')) + '\\weights.cpkt'
network.save_weights(weight_store_path)
del network
print('saved to weight.ckpt')


network = MyNetwork()
network.compile(keras.optimizers.Adam(lr=1e-3),
                loss=tf.losses.CategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
network.load_weights(weight_store_path)
print('load weight')
network.evaluate(db_val)