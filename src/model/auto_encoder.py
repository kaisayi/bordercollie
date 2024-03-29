#!/usr/bin/python
# -*-coding:utf8-*-

"""
@author: LieOnMe
@time: 2019/7/30 22:33
"""

import os
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow import keras
import matplotlib.pyplot as plt
from utils import conf

tf.random.set_seed(22)
np.random.seed(22)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
assert tf.__version__.startswith('2.')
PROJ_PATH = conf.get_project_path()


def save_image(imgs, name):
    new_im = Image.new('L', (280, 280))

    index = 0
    for i in range(0, 280, 28):
        for j in range(0, 280, 28):
            im = imgs[index]
            im = Image.fromarray(im, mode='L')
            new_im.paste(im, (i, j))
            index += 1
    new_im.save(name)


h_dim = 20
batch_size = 128
lr = 1e-3

(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
x_train, x_test = x_train.astype(np.float32) / 255, x_test.astype(np.float32) / 255

# do not need label
train_db = tf.data.Dataset.from_tensor_slices(x_train)
train_db = train_db.shuffle(10000).batch(batch_size)
test_db = tf.data.Dataset.from_tensor_slices(x_test)
test_db = test_db.batch(batch_size)

print(x_train.shape)
print(x_test.shape)
out_path = os.path.normpath(os.path.join(PROJ_PATH, 'out/fashion/'))


class AE(keras.Model):

    def __init__(self):
        super(AE, self).__init__()

        # Encoder
        self.encoder = keras.Sequential([
            keras.layers.Dense(256, activation=tf.nn.relu),
            keras.layers.Dense(128, activation=tf.nn.relu),
            keras.layers.Dense(h_dim),
        ])

        # Decoder
        self.decoder = keras.Sequential([
            keras.layers.Dense(128, activation=tf.nn.relu),
            keras.layers.Dense(256, activation=tf.nn.relu),
            keras.layers.Dense(784),
        ])

    def call(self, inputs, training=None, mask=None):
        # [b, 784] => [b, h]
        h = self.encoder(inputs)
        # [b, h] => [b, 784]
        x_hat = self.decoder(h)

        return x_hat


model = AE()
model.build(input_shape=(None, 784))
optimizer = keras.optimizers.Adam(lr=lr)

for epoch in range(100):
    for step, x in enumerate(train_db):
        # flatten
        x = tf.reshape(x, [-1, 784])

        with tf.GradientTape() as tape:
            x_rec_logits = model(x)
            rec_loss = tf.losses.binary_crossentropy(x, x_rec_logits, from_logits=True)
            rec_loss = tf.reduce_mean(rec_loss)

        grads = tape.gradient(rec_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if step % 100 == 0:
            print("Epoch: {}, step: {}, loss: {}"
                  .format(epoch, step, float(rec_loss)))

    # evaluation
    x = next(iter(test_db))
    logits = model(tf.reshape(x, [-1, 784]))
    x_hat = tf.sigmoid(logits)

    # [b, 784] => [b, 28, 28]
    x_hat = tf.reshape(x_hat, [-1, 28, 28])

    # concat
    # x_concat = tf.concat([x, x_hat], axis=0)
    x_concat = x_hat.numpy() * 255
    x_concat = x_concat.astype(np.uint8)
    save_image(x_concat, out_path + '\\ae_image_epoch_%d.png'%epoch)
