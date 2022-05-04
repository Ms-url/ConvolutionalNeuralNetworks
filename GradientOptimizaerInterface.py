# 2022.3.4
# YiMingLi

import numpy as np

'''
梯度下降优化方法类
    超参数 v, cache 初始化为 0  
        Nesterov-改进动量法
        Adam-自适应矩估计
        AmsGrad-Adam变体
'''
class GradientOptimizerInterface():
    Optimizers = ['nesterov', 'adam', 'amsGrad']

    @staticmethod
    def optimizerInterface(optimizer,lr, x, dx, v, mu = 0.9, cache = 0, t=1):
        if optimizer == 'nesterov':
            return GradientOptimizerInterface.nesterov(lr, x, dx, v, mu , cache , t)
        elif optimizer == 'adam':
            return GradientOptimizerInterface.adam(lr, x, dx, v, mu , cache , t)
        elif optimizer == 'amsGrad':
            return GradientOptimizerInterface.amsGrad(lr, x, dx, v, mu , cache , t)

    # lr: 学习率
    # x: 参数
    # dx: 参数梯度
    # v: 下降速度 初始化0
    # mu: 摩擦系数 0.9 0.99 0.5
    # return: 更新率
  
    @staticmethod
    def nesterov(lr, x, dx, v, mu = 0.9, cache = 0, t=1):
        eps = 1e-8
        pre_v = v
        v = mu*v -lr*dx
        update = v + mu*(v - pre_v)
        update_ratio = np.sum(np.abs(update)) / np.sum(np.abs(x)+eps)
        x += update
        return update_ratio, x, v

    @staticmethod
    def adam(lr, x, dx, v, mu = 0.9, cache = 0, t=1):
        # t 训练次数
        # cache 初始化0
        decay_rate = 0.999
        eps = 1e-8

        v = mu*v + (1-mu)*dx
        vt = v/(1-mu**t) 

        cache = decay_rate*cache + (1-decay_rate)*(dx**2)
        cachet = cache/(1-decay_rate**t)

        update = -((lr/np.sqrt(cachet)+eps))*vt 
        update_ratio = np.sum(np.abs(x)) / np.sum(np.abs(update))

        x += update 
        return  update_ratio, x, v

    @staticmethod
    def amsGrad(lr, x, dx, v, mu = 0.9, cache = 0, t=1):
        # t 训练次数
        # cache 初始化0
        decay_rate = 0.999
        eps = 1e-8

        v = mu*v + (1-mu)*dx
        vt = v/(1-mu**t) 

        # 这一步是和 Adam 唯一的不同
        cache = np.max(cache , decay_rate*cache + (1-decay_rate)*(dx**2))

        cachet = cache/(1-decay_rate**t)

        update = -((lr/np.sqrt(cachet)+eps))*vt 
        update_ratio = np.sum(np.abs(x)) / np.sum(np.abs(update))

        x += update 
        return  update_ratio, x, v 
    


    @staticmethod
    def check_optimizer(optimizer):
        if optimizer not in GradientOptimizerInterface.Optimizers:
            raise ValueError(f"Optimizer error : Only {GradientOptimizerInterface.Optimizers} functions are encapsulated")
        pass


