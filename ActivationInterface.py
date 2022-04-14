# 2022/3/4
# YiMingLi

import numpy as np
'''
激活函数类
'''
class ActivationInterface(object):
    activations = ["ReLU"]

    @staticmethod
    def activation(data, activation = "ReLU"):
        if activation=="ReLU":
            data = np.maximum(0,data)
            return data
        pass

    @staticmethod
    def d_activation(d_data,data,activation = "ReLU"):
        if activation=="ReLU":
            d_data[data<=0] = 0
            return d_data 
        pass

    @staticmethod
    def check_activation(activation):
        if activation not in ActivationInterface.activations:
            raise ValueError(f"activation error : Only {ActivationInterface.activations} functions are encapsulated")


