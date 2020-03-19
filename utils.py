#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time    : 2020-03-19 22:18
@Author  : Wang Xin
@Email   : wangxin_buaa@163.com
@File    : utils.py.py
"""


import numpy as np


# 距离函数
def distance(x, y, p=2):
    """
    :param x: numpy.array (d)
    :param y: numpy.array (c,d)
    :param p: int, norm type
    :return d: numpy.array (c,)
    """
    tmp = np.sum((x-y)**p, axis=1)
    return np.sqrt(tmp)