#!/usr/bin/python
#-*-coding:utf8-*-

"""
@author: LieOnMe
@time: 2019/7/18 22:30
"""

import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print('GPU', tf.test.is_gpu_available())

a = tf.constant(2.)
b = tf.constant(4.)

print(a * b)
