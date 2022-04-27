import struct

from VGG import VGG

from CNNtrainInterface import CNNtarinInterface

from MNISTInterface import MNISTInterface

class VGGTest(VGG,  CNNtarinInterface,  MNISTInterface):
    pass


if __name__ == '__main__':

    print('-------------------------------------start')

    struct = [] # 线性模型
    # struct = ['FC_64'] # 只含一个隐含层
    # struct = ['conv_6_5_1_1'] + ['pool'] + ['conv_16_5_1_1'] + ['pool'] + ['FC_120'] + ['FC_84']  # LeNet
    # struct = ['conv_8'] + ['pool'] + ['conv_12']*3 + ['pool'] + ['conv_36']*3 + ['pool'] + ['FC_64'] # VGG

    vgg = VGGTest(struct)
    num_samples_ratio = 0.7 # 训练数据百分比

    vgg.load_train_data(num_samples_ratio)

    epoch_more = 1
    lr = 10**-3
    reg = 10**-4
    batch = 64 
    lr_decay = 0.8 
    mu = 0.9 
    optimizer = 'nesterov'
    regularization = "L2"
    activation = "ReLU"

    vgg.train(epoch_more ,  lr = 10**-3,  reg = 10**-4,batch = 64, lr_decay = 0.8, mu = 0.9, 
                optimizer = 'nesterov', regularization = "L2", activation = "ReLU")

    print('-----------------------------------------end')
    