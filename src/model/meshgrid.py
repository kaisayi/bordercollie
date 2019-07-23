#!/usr/bin/python
# -*-coding:utf8-*-

"""
@author: LieOnMe
@time: 2019/7/23 21:53
"""
import tensorflow as tf
import matplotlib.pyplot as plt


def func(x):
    """
    :param x: [b, 2]
    :return:
    """
    z = tf.math.sin(x[...,0]) + tf.math.sin(x[...,1])
    return z


x = tf.linspace(0., 2 * 3.14, 500)
y = tf.linspace(0., 2 * 3.14, 500)

points_x, points_y = tf.meshgrid(x, y)

points = tf.stack([points_x, points_y], axis=2)
print("Shape of points: ", points.shape)

z = func(points)
print("shape of z: ", z.shape)

plt.figure('plot 2d func value')
plt.imshow(z, origin='lower', interpolation='none')
plt.colorbar()

plt.figure('plot 2d func contour')
plt.contour(points_x, points_y, z)
plt.colorbar()

plt.show()

