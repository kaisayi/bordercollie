#!/usr/bin/python
# -*-coding:utf8-*-

"""
@author: LieOnMe
@time: 2019/7/26 0:37
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


def himmelblau(x):
    return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2


def visualize():
    x = np.arange(-6, 6, 0.1)
    y = np.arange(-6, 6, 0.1)
    print('x, y range: ', x.shape, y.shape)
    X, Y = np.meshgrid(x, y)
    print('X, Y range: ', X.shape, Y.shape)
    Z = himmelblau([X, Y])

    fig = plt.figure('Himmelblau')
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Z)
    ax.view_init(60, -30)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.savefig("D:\\code\\workspace-self\\tensorflow-pg\\out\\him.png")


def train_op():
    x = tf.constant([-4., 0.])

    for step in range(200):
        with tf.GradientTape() as tape:
            tape.watch([x])
            y = himmelblau(x)
        grads = tape.gradient(y, [x])[0]
        x -= 0.01 * grads

        if step % 20 == 0:
            print('step {} : x = {}, f(x) = {}'
                  .format(step, x.numpy(), y.numpy()))


if __name__ == "__main__":
    # train_op()
    # visualize()
    import os
    print(os.path.dirname(__file__))
