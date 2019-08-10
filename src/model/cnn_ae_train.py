#!/usr/bin/python
# -*-coding:utf8-*-

"""
@author: LieOnMe
"""

import glob
import os

import numpy as np
import tensorflow as tf
from PIL import Image
from utils import conf
from utils.dataset import make_chinese_char_dataset
from tensorflow import keras

tf.random.set_seed(22)
np.random.seed(22)
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# assert tf.__version__.startswith('2.')
PROJ_PATH = conf.get_project_path()

validate_size = 5000
batch_size = 128
epoches = 100
LATENT_NUM = 128
img_path = glob.glob(os.path.normpath(os.path.join(PROJ_PATH, "data/chinese-char/img/*.jpg")))
np.random.shuffle(img_path)
train_img_path = img_path[validate_size:]
test_img_path = img_path[:validate_size]
train_db, img_shape, _ = make_chinese_char_dataset(train_img_path, batch_size, repeat=32)
test_db, _, _ = make_chinese_char_dataset(test_img_path, batch_size)
out_path = os.path.normpath(os.path.join(PROJ_PATH, 'out/chinese-char/'))
print(img_shape)


def save_image(imgs, name):
    new_im = Image.new('L', (320, 320))

    index = 0
    for i in range(0, 320, 32):
        for j in range(0, 320, 32):
            im = imgs[index]
            im = Image.fromarray(im, mode='L')
            new_im.paste(im, (i, j))
            index += 1
    new_im.save(name)


class CNN_AE(keras.Model):

    def __init__(self):
        super(CNN_AE, self).__init__()

        # Encoder
        self.encoder = keras.Sequential([
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(64),
        ])

        # Decoder
        self.decoder = keras.Sequential([
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(img_shape[0] * img_shape[1]),
        ])

    def call(self, inputs, training=None, mask=None):
        # encode [b, 32 * 32] => [b, 64]
        h = self.encoder(inputs)

        # decode [b, 64] => [b, 32 * 32]
        x_hat = self.decoder(h)
        return x_hat


cnnae = CNN_AE()
cnnae.build(input_shape=(None, 32 * 32))
optimizer = keras.optimizers.Adam(lr=1e-3)
cnnae.summary()

for step, x in enumerate(train_db):
    x = tf.reshape(x, [-1, 32 * 32])
    with tf.GradientTape() as tape:
        x_logit = cnnae(x)
        bi_loss = tf.losses.binary_crossentropy(x, x_logit, from_logits=True)
        bi_loss = tf.reduce_mean(bi_loss)

    grads = tape.gradient(bi_loss, cnnae.trainable_variables)
    optimizer.apply_gradients(zip(grads, cnnae.trainable_variables))

    if step % 100 == 0:
        print("Step: {}, loss: {}"
              .format(step, float(bi_loss)))

    if step % 500 == 0:
        x_test = next(iter(test_db))
        test_out = cnnae(tf.reshape(x_test, [-1, 32 * 32]))
        x_hat = tf.sigmoid(test_out)

        x_hat = tf.reshape(x_hat, [-1, 32, 32])

        x_hat = x_hat.numpy() * 255
        x_hat = x_hat.astype(np.uint8)
        save_image(x_hat, out_path + '\\new_ae_image_test_%d.png' % step)
