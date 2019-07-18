#!/usr/bin/python
# -*-coding:utf8-*-

"""
@author: LieOnMe
@time: 2019/7/18 22:45
"""

import numpy as np
import pandas as pd

# initialize value
NUM_POINTS = 1000
data = pd.DataFrame({'x': [np.random.normal(9.0, 10) for _ in range(NUM_POINTS)]})
data['y'] = data['x'].apply(lambda x: x * 0.423 - 5.32 + np.random.normal(0.0, 0.03))


# calculate loss
def compute_error_for_line_points(b, w, points):
    total_error = 0.0
    for i in range(len(points)):
        _x = points[i, 0]
        _y = points[i, 1]
        total_error += (_y - (w * _x + b)) ** 2
    return total_error / len(points)


# calculate gradient and update
def step_gradient(b_cur, w_cur, points, lr):
    """
    :param b_cur: current b
    :param w_cur: current slope
    :param points: collections of points
    :param lr: learning rate
    :return: [new_b, new_w]
    """
    b_gradient = 0.0
    w_gradient = 0.0
    N = len(points)
    for i in range(len(points)):
        _x = points[i, 0]
        _y = points[i, 1]
        # grad_b = 2(wx + b - y)
        b_gradient += (2 / N) * (_x * w_cur + b_cur - _y)
        w_gradient += (2 / N) * (_x * w_cur + b_cur - _y) * _x
    # update w & b
    new_b = b_cur - (lr * b_gradient)
    new_w = w_cur - (lr * w_gradient)
    return [new_b, new_w]


def gradient_descent_runner(points, start_b, start_w, lr, num_iterations):
    b = start_b
    w = start_w
    # update for several times
    for i in range(num_iterations):
        b, w = step_gradient(b, w, points, lr)
    return [b, w]


def run():
    points = data.values
    print("Data-info for training:\n {}".format(points[:5, :]))
    lr = 0.001
    init_b = init_w = 0.0
    num_iterations = 2000
    print("Starting gradient descent at b = {0}, w = {1}, error = {2}"
          .format(init_b, init_w, compute_error_for_line_points(init_b, init_w, points)))
    print("Running...")
    [b, w] = gradient_descent_runner(points, init_b, init_w, lr, num_iterations)
    print("After {0} iterations b = {1}, w = {2}, error = {3}"
          .format(num_iterations, b, w, compute_error_for_line_points(b, w, points)))


if __name__ == "__main__":
    run()
