# 2022/3/4
# YiMingLi

from ast import Param
import numpy as np
'''
范数正则化类
正则化技术：在损失函数中增加对模型的偏好 reg_loss
'''
class RegularizationInterface(object):
    regularizations = ['L1','L2']

    # weight: 权重 
    # reg: 正则化系数 
    # regularization: 正则化类型 

    @staticmethod
    def norm_regularization(weight, regularization, reg=10**-4):
        if regularization=='L1':
            return np.sum(np.abs(weight))*reg

        elif regularization=='L2':
            return np.sum(weight*weight)*reg/2

    @staticmethod
    def d_norm_regularization(weight , regularization, reg=10**-4):
        if regularization=='L1':
            return np.sign(weight)*reg

        elif regularization=='L2':
            return weight*reg

    @staticmethod
    def check_norm_regularization(regularization):
        if regularization not in RegularizationInterface.regularizations:
            raise ValueError(f"Regularization error: Only {RegularizationInterface.regularizations} are encapsulated")


