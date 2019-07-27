#!/usr/bin/python
# -*-coding:utf8-*-

"""
@author: LieOnMe
@time: 2019/7/27 14:13
"""
import os


def get_project_path():
    dir_name = os.path.dirname(__file__)
    return os.path.normpath(os.path.join(dir_name, "../../"))


def get_src_path():
    dir_name = os.path.dirname(__file__)
    return os.path.normpath(os.path.join(dir_name, "../"))
