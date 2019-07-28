#!/usr/bin/python
# -*-coding:utf8-*-

"""
@author: LieOnMe
@time: 2019/7/29 0:44
"""

import tensorflow as tf
from tensorflow import keras


class BasicBlock(keras.layers.Layer):

    def __init__(self, filter_num, strides=1):
        super(BasicBlock, self).__init__()

        self.conv1 = keras.layers.Conv2D(filter_num, kernel_size=(3, 3), strides=strides, padding='same')
        # BatchNormalization 参考 https://www.cnblogs.com/guoyaohua/p/8724433.html
        self.bn1 = keras.layers.BatchNormalization()
        self.relu = keras.layers.Activation('relu')
        self.conv2 = keras.layers.Conv2D(filter_num, kernel_size=(3, 3), strides=1, padding='same')
        self.bn2 = keras.layers.BatchNormalization()

        if strides != 1:
            self.down_sample = keras.Sequential()
            self.down_sample.add(keras.layers.Conv2D(filter_num, kernel_size=(1, 1), strides=strides))
            self.down_sample.add(keras.layers.BatchNormalization())
        else:
            self.down_sample = lambda x: x

        self.strides = strides

    def call(self, inputs, training=None):
        residual = self.down_sample(inputs)

        conv1 = self.conv1(inputs)
        bn1 = self.bn1(conv1)
        relu1 = self.relu(bn1)
        conv2 = self.conv2(relu1)
        bn2 = self.bn2(conv2)

        add = keras.layers.add([bn2, residual])
        out = self.relu(add)
        return out

    def compute_output_signature(self, input_signature):
        pass


class ResNet(keras.Model):

    def __init__(self, layer_dims, num_classes=100):
        """
        :param layer_dims: [2, 2, 2, 2] resnet18
        :param num_classes:
        """
        super(ResNet, self).__init__()

        self.stem = keras.Sequential([
            keras.layers.Conv2D(64, kernel_size=(2, 2), strides=(1, 1)),
            keras.layers.BatchNormalization(),
            keras.layers.Activation('relu'),
            keras.layers.MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding='same')
        ])

        self.layer1 = self.build_res_block(64, layer_dims[0])
        self.layer2 = self.build_res_block(128, layer_dims[1], strides=2)
        self.layer3 = self.build_res_block(256, layer_dims[2], strides=2)
        self.layer4 = self.build_res_block(512, layer_dims[3], strides=2)

        # output: [b, 512, h, w] => [b, 512]
        self.avgpool = keras.layers.GlobalAveragePooling2D()
        self.fc = keras.layers.Dense(num_classes)

    def call(self, inputs, training=None, mask=None):
        x = self.stem(inputs)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.fc(x)

        return x

    def build_res_block(self, filter_num, blocks, strides=1):
        res_blocks = keras.Sequential()

        # may down sample
        res_blocks.add(BasicBlock(filter_num, strides))

        for _ in range(1, blocks):
            res_blocks.add(BasicBlock(filter_num, strides=1))

        return res_blocks


def resnet18():
    return ResNet([2, 2, 2, 2])


def resnet34():
    return ResNet([3, 4, 6, 3])
