#!/usr/bin/python
# -*-coding:utf8-*-

"""
@author: LieOnMe
@time: 2019/7/27 15:54
"""
import os
import tensorflow as tf
from tensorflow import keras

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def pre_process(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255
    y = tf.cast(y, dtype=tf.int32)
    return x, y


(x, y), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
batch_size = 64
db = tf.data.Dataset.from_tensor_slices((x, y))
db = db.map(pre_process).shuffle(60000).batch(batch_size).repeat(10)
ds_val = tf.data.Dataset.from_tensor_slices((x_test, y_test))
ds_val = ds_val.map(pre_process).batch(batch_size)


class MyDense(keras.layers.Layer):

    def __init__(self, inp_dim, outp_dim):
        super(MyDense, self).__init__()

        self.kernel = self.add_variable('w', [inp_dim, outp_dim])
        self.bias = self.add_variable('b', [outp_dim])

    def call(self, inputs, **kwargs):
        out = inputs @ self.kernel + self.bias
        return out


class MyModel(keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = MyDense(28 * 28, 256)
        self.fc2 = MyDense(256, 128)
        self.fc3 = MyDense(128, 64)
        self.fc4 = MyDense(64, 32)
        self.fc5 = MyDense(32, 10)

    def call(self, inputs, training=None, mask=None):
        x = self.fc1(inputs)
        x = tf.nn.relu(x)
        x = self.fc2(x)
        x = tf.nn.relu(x)
        x = self.fc3(x)
        x = tf.nn.relu(x)
        x = self.fc4(x)
        x = tf.nn.relu(x)
        x = self.fc5(x)
        return x


network = MyModel()

network.compile(optimizer=keras.optimizers.Adam(lr=0.01),
                loss=tf.losses.CategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

network.fit(db, epochs=5, validation_data=ds_val, validation_freq=2)
network.evaluate(ds_val)
