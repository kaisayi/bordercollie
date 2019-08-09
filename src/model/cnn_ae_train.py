#!/usr/bin/python
# -*-coding:utf8-*-

"""
@author: LieOnMe
@time: 2019/8/9 21:35
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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
assert tf.__version__.startswith('2.')
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
    new_im = Image.new('L', (160, 160))

    index = 0
    for i in range(0, 160, 32):
        for j in range(0, 160, 32):
            im = imgs[index]
            im = Image.fromarray(im, mode='L')
            new_im.paste(im, (i, j))
            index += 1
    new_im.save(name)


class CNN_AE(keras.Model):

    def __init__(self):
        super(CNN_AE, self).__init__()

        # Encoder
        self.conv1 = keras.layers.Conv2D(filters=32, kernel_size=3, strides=2, padding='same')  # 16, 16, 32
        self.conv2 = keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, padding='same')  # 8, 8, 64
        self.bn2 = keras.layers.BatchNormalization()
        self.conv3 = keras.layers.Conv2D(filters=128, kernel_size=3, strides=2, padding='same')  # 4, 4, 128
        self.bn3 = keras.layers.BatchNormalization()
        self.flat = keras.layers.Flatten()
        self.fc1 = keras.layers.Dense(256)
        self.bn1 = keras.layers.BatchNormalization()
        self.latent = keras.layers.Dense(LATENT_NUM)

        # Decoder
        self.fc2 = keras.layers.Dense(256, activation='relu')
        self.fc3 = keras.layers.Dense(128 * 4 * 4)
        self.bn6 = keras.layers.BatchNormalization()
        self.reshape = keras.layers.Reshape((4, 4, 128))
        self.deconv1 = keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same')  # 8, 8, 64
        self.bn4 = keras.layers.BatchNormalization()
        self.deconv2 = keras.layers.Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same')  # 16, 16, 32
        self.bn5 = keras.layers.BatchNormalization()
        self.deconv3 = keras.layers.Conv2DTranspose(filters=1, kernel_size=3, strides=2, padding='same')  # 32, 32, 1

    def call(self, inputs, training=None, mask=None):
        # encode
        x = tf.nn.leaky_relu(self.conv1(inputs))
        x = tf.nn.leaky_relu(self.bn2(self.conv2(x), training=training))
        x = tf.nn.leaky_relu(self.bn3(self.conv3(x), training=training))

        x = self.flat(x)
        x = tf.nn.leaky_relu(self.bn1(self.fc1(x), training=training))
        z = self.latent(x)

        # decode
        x = tf.nn.relu(self.fc2(z))
        x = tf.nn.relu(self.bn6(self.fc3(x), training=training))
        x = self.reshape(x)
        x = tf.nn.relu(self.bn4(self.deconv1(x), training=training))
        x = tf.nn.relu(self.bn5(self.deconv2(x), training=training))
        out = tf.nn.sigmoid(self.deconv3(x))

        return out


cnnae = CNN_AE()
cnnae.build(input_shape=(None, 32, 32, 1))
optimizer = keras.optimizers.Adam(lr=1e-3)
cnnae.summary()

for step, x in enumerate(train_db):
    with tf.GradientTape() as tape:
        x_out = cnnae(x)
        bi_loss = tf.losses.binary_crossentropy(x, x_out)
        bi_loss = tf.reduce_mean(bi_loss)

    grad = tape.gradient(bi_loss, cnnae.trainable_variables)
    optimizer.apply_gradients(zip(grad, cnnae.trainable_variables))

    if step % 100 == 0:
        print("Step: {}, loss: {}"
              .format(step, float(bi_loss)))

    if step % 500 == 0:
        x_test = next(iter(test_db))
        test_out = cnnae(x_test)

        test_out = test_out.numpy() * 255
        test_out = test_out.astype(np.uint8)
        test_out = np.squeeze(test_out, axis=3)
        save_image(test_out, out_path + '\\ae_image_test_%d.png' % step)
