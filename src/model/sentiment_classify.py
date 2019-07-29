#!/usr/bin/python
# -*-coding:utf8-*-

"""
@author: LieOnMe
@time: 2019/7/29 22:46
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

tf.random.set_seed(22)
np.random.seed(22)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
assert tf.__version__.startswith('2.')

batch_size = 64
embedding_len = 100
units = 64
# the most frequent words
total_words = 10000
max_review_len = 80
(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=total_words)
# [b, seq_len] => [b, 80]
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_review_len)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_review_len)

db_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
db_train = db_train.shuffle(1000).batch(batch_size, drop_remainder=True)
db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
db_test = db_test.batch(batch_size, drop_remainder=True)

print('x_train.shape: ', x_train.shape, tf.reduce_max(y_train), tf.reduce_min(y_train))
print('x_test.shape: ', x_test.shape)


class MyRNN(keras.Model):

    def __init__(self):
        super(MyRNN, self).__init__()

        self.state0 = [tf.zeros([batch_size, units])]
        # transform text to embedding representation
        # [b, 80] => [b, 80, 100]
        self.embedding = keras.layers.Embedding(total_words, embedding_len, input_length=max_review_len)

        # rnn cells
        self.rnn_cell0 = keras.layers.SimpleRNNCell(units, dropout=0.2)

        # fc
        self.fc = keras.layers.Dense(1)

    def call(self, inputs, training=None, mask=None):
        """
        :param inputs: [b, 80]
        :param training:  train or test for dropout
        :param mask:
        :return:
        """
        x = self.embedding(inputs)
        # rnn cell
        # [b, 80, 100] => [b, 64]
        state0 = self.state0
        out = self.state0[0]
        for word in tf.unstack(x, axis=1):
            # x*wxh + h * whh
            out, state1 = self.rnn_cell0(word, state0, training)
            state0 = state1
        # out: [b, 64] => [b, 1]
        x = self.fc(out)
        prob = tf.sigmoid(x)
        return prob


def main():
    epochs = 4
    model = MyRNN()
    model.compile(optimizer=keras.optimizers.Adam(0.001),
                  loss=tf.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])
    model.fit(db_train, epochs=epochs, validation_data=db_test)

    model.evaluate(db_test)


if __name__ == '__main__':
    main()
