#!/usr/bin/python
# -*-coding:utf8-*-

"""
@author: LieOnMe
@time: 2019/7/27 16:37
"""

import os
import tensorflow as tf
from tensorflow import keras
from utils import conf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
PROJ_PATH = conf.get_project_path()


def pre_process(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255
    x = tf.reshape(x, [28 * 28])
    y = tf.cast(y, dtype=tf.int32)
    y = tf.one_hot(y, depth=10)
    return x, y


(x, y), (x_val, y_val) = keras.datasets.mnist.load_data(path="D:\\code\\workspace-self\\tensorflow-pg\\data\\mnist"
                                                             "\\mnist.npz")

batch_size = 64
db = tf.data.Dataset.from_tensor_slices((x, y))
db = db.map(pre_process).shuffle(10000).batch(batch_size)
ds_val = tf.data.Dataset.from_tensor_slices((x_val, y_val))
ds_val = ds_val.map(pre_process).batch(batch_size)

network = keras.Sequential([
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(10)
])

network.build(input_shape=(None, 28 * 28))
network.summary()

network.compile(optimizer=keras.optimizers.Adam(lr=0.01),
                loss=tf.losses.CategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
network.fit(db, epochs=3, validation_data=ds_val, validation_freq=2)

network.evaluate(ds_val)

# weight_store_path = os.path.normpath(os.path.join(PROJ_PATH, 'out/mnist')) + '\\weights.cpkt'
model_store_path = os.path.normpath(os.path.join(PROJ_PATH, 'out/mnist')) + '\\model.h5'
# network.save_weights(weight_store_path)
network.save(model_store_path)
# print("Saved weights")
print("Saved whole model")
del network

# network = keras.Sequential([
#     keras.layers.Dense(256, activation='relu'),
#     keras.layers.Dense(128, activation='relu'),
#     keras.layers.Dense(64, activation='relu'),
#     keras.layers.Dense(32, activation='relu'),
#     keras.layers.Dense(10)
# ])
# network.compile(optimizer=keras.optimizers.Adam(lr=0.01),
#                 loss=tf.losses.CategoricalCrossentropy(from_logits=True),
#                 metrics=['accuracy'])

# network.load_weights(weight_store_path)
network = keras.models.load_model(model_store_path)
# print("Loaded weights")
print('loaded model')
network.evaluate(ds_val)

# 更通用的保存方式，可以给其他的语言使用
tf.saved_model.save(network, '/tmp/saved_model/')

imported = tf.saved_model.load('/tmp/saved_model/')
f = imported.signatures['serving_default']
print(f(x=tf.ones([1, 28, 28, 3])))