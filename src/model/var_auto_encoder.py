#!/usr/bin/python
# -*-coding:utf8-*-

"""
@author: LieOnMe
@time: 2019/7/30 23:31
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

z_dim = 10


class VAE(keras.Model):

    def __init__(self):
        super(VAE, self).__init__()

        # Encoder
        self.fc1 = keras.layers.Dense(128)
        self.fc2 = keras.layers.Dense(z_dim)  # get a mean prediction
        self.fc3 = keras.layers.Dense(z_dim)  # get a std prediction

        # Decoder
        self.fc4 = keras.layers.Dense(128)
        self.fc5 = keras.layers.Dense(784)

    def reparameterize(self, mu, log_var):
        eps = tf.random.normal(log_var.shape)
        std = tf.exp(log_var) ** 0.5
        z = mu + std * eps
        return z

    def call(self, inputs, training=None, mask=None):
        mu, log_var = self.encoder(inputs)

        # reparameterization trick
        z = self.reparameterize(mu, log_var)

        x_hat = self.decoder(z)
        return x_hat, mu, log_var

    def encoder(self, x):
        h = tf.nn.relu(self.fc1(x))
        # get mean
        mu = self.fc2(h)
        # get variance
        log_var = self.fc3(h)
        return mu, log_var

    def decoder(self, z):
        out = tf.nn.relu(self.fc4(z))
        out = self.fc5(out)

        return out


model = VAE()
model.build(input_shape=(None, 784))
optimizer = keras.optimizers.Adam(lr=lr)

for epoch in range(100):
    for step, x in enumerate(train_db):
        # flatten
        x = tf.reshape(x, [-1, 784])
        with tf.GradientTape() as tape:
            x_rec_logits, mu, log_var = model(x)
            rec_loss = tf.losses.binary_crossentropy(x, x_rec_logits, from_logits=True)
            rec_loss = tf.reduce_mean(rec_loss)

            # compute kl divergence (mu, var) ~ N(0, 1)
            kl_div = -0.5 * (log_var + 1 - mu ** 2 - tf.exp(log_var))
            kl_div = tf.reduce_mean(kl_div)

            loss = rec_loss + 1. * kl_div
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(loss, model.trainable_variables))

        if step % 100 == 0:
            print("Epoch: {}, step: {}, rec_loss: {}, kl_loss: {}"
                  .format(epoch, step, float(rec_loss), float(kl_div)))
