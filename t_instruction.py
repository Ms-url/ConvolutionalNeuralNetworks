# 2022/4/25
# YiMingLi

import struct
from CNNtrainInterface import CNNtarinInterface
# 模型类
from VGG import VGG
# 数据载入类
from MNISTInterface import MNISTInterface
# 创建顶层对象类 继承 训练类，模型类，数据载入类
class VGGTest(VGG,  CNNtarinInterface,  MNISTInterface):
    pass

'''
CNNblockInterface -- 卷积网络基本模块类
    封装卷积层、池化层、全连接层、softmax损失函数 的向前和反向计算，以及参数初始化接口 
CNNtrainInterface -- 网络训练类
    封装训练，随机搜索训练，测试三个接口

VGG -- 网络模型类 继承 CNNblockInterface 
    __init__(struct)
        设置结构, last FC 不用定义在 strcut 中

    解析网络结构，并进行
    init_params             各层网络权重和偏置初始化
    forward                 向前传播过程        
    backpropagation         反向传播过程        
    params_updata           参数更新过程
    封装接口：
        save_checkpoint         保存模型
        load_checkpoint         载入保存的模型

MNISTInterface -- 数据载入类
    set_data_pro            数据集超参数设置
    load_train_data         载入训练数据        
    load_test_data          载入测试数据
    数据载入
        self.num_samples = labels.size     
        self.num_train_samples
        self.num_val_samples
        self.train_data
        self.train_labels
        self.val_data
        self.val_labels         
        self.test_data          
        self.test_labels       
        self.num_class          
        self.im_height 
        self.im_width 
        self.im_dims               

'''
