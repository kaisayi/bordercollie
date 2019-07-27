#!/usr/bin/python
# -*-coding:utf8-*-

"""
@author: LieOnMe
@time: 2019/7/27 11:57
"""
import os
from datetime import datetime
import tensorflow as tf
from tensorflow import keras
from utils import conf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
PROJ_PATH = conf.get_project_path()


def pre_process(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255
    y = tf.cast(y, dtype=tf.int32)
    return x, y


(x, y), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
print(x.shape, y.shape)
print(x_test.shape, y_test.shape)
batch_size = 64

db = tf.data.Dataset.from_tensor_slices((x, y))
db = db.map(pre_process).shuffle(60000).batch(batch_size).repeat(10)

db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
db_test = db_test.map(pre_process).batch(batch_size)

db_iter = iter(db)
sample = next(db_iter)
print('batch : ', sample[0].shape, sample[1].shape)

model = keras.Sequential([
    keras.layers.Dense(256, activation=tf.nn.relu),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(10)
])

model.build(input_shape=[None, 28 * 28])
model.summary()
optimizer = keras.optimizers.Adam(lr=1e-2)

# create meter
acc_meter = keras.metrics.Accuracy()
loss_meter = keras.metrics.Mean()

current_time = datetime.now().strftime("%Y%m%d-%H%M")
log_dir = os.path.normpath(os.path.join(PROJ_PATH, "out/logs")) + '\\' + current_time
summary_writer = tf.summary.create_file_writer(log_dir)

sample_img = sample[0]
sample_img = tf.reshape(sample_img[0], [1, 28, 28, 1])

with summary_writer.as_default():
    tf.summary.image("Training sample: ", sample_img, step=0)


def main():
    for step, (_x, _y) in enumerate(db):
        _x = tf.reshape(_x, [-1, 28 * 28])
        y_onehot = tf.one_hot(_y, depth=10)
        with tf.GradientTape() as tape:
            logits = model(_x)
            loss_mse = tf.reduce_mean(tf.losses.MSE(y_onehot, logits))
            loss_ce = tf.reduce_mean(tf.losses.categorical_crossentropy(y_onehot, logits, from_logits=True))
            loss_meter.update_state(loss_ce)

        grads = tape.gradient(loss_ce, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if step % 100 == 0:
            print("Step: {}, loss_mse : {}, loss_ce : {}"
                  .format(step, float(loss_mse), loss_meter.result().numpy()))
            loss_meter.reset_states()

            with summary_writer.as_default():
                tf.summary.scalar('train loss', float(loss_ce), step=step)

        # evaluate
        if step % 500 == 0:
            total_num = 0
            total_correct = 0
            acc_meter.reset_states()
            for _xt, _yt in db_test:
                _xt = tf.reshape(_xt, [-1, 28 * 28])
                logits = model(_xt)
                # logits => prob
                prob = tf.nn.softmax(logits, axis=1)
                # [b, 10] => [b]
                pred = tf.cast(tf.argmax(prob, axis=1), dtype=tf.int32)
                correct = tf.equal(pred, _yt)
                correct = tf.reduce_sum(tf.cast(correct, dtype=tf.int32))
                total_correct += int(correct)
                total_num += _xt.shape[0]

                acc_meter.update_state(_yt, pred)

            acc = total_correct / total_num
            print("Test Acc : {}"
                  .format(acc_meter.result().numpy()))

            val_images = x[:25]
            val_images = tf.reshape(val_images, [-1, 28, 28, 1])
            with summary_writer.as_default():
                tf.summary.scalar('test_acc', float(acc), step=step)
                tf.summary.image('val-onebyone-images:', val_images, max_outputs=25, step=step)


if __name__ == '__main__':
    main()
