import struct

from VGG import VGG

from CNNtrainInterface import CNNtarinInterface

from MNISTInterface import MNISTInterface

class VGGTest(VGG,  CNNtarinInterface,  MNISTInterface):
    pass


if __name__ == '__main__':

    print('-------------------------------------start')

    struct = [] # 线性模型
    # struct = ['FC_64']+['FC_128'] # 前馈神经网络
    # struct = ['conv_6_5_1_2'] + ['pool'] + ['conv_16_5_1_0'] + ['pool'] + ['FC_120'] + ['FC_84']  # LeNet
   
    # struct = ['conv_64']*2 + ['pool'] + ['conv_128']*2 + ['pool'] + ['conv_256']*3 + ['pool'] \
    #        + ['conv_512']*3 + ['pool'] + ['conv_512']*3 + ['pool'] + ['FC_4096'] + ['FC_4096'] # VGG 

    # struct = ['conv_8'] + ['pool'] + ['conv_12']*3 + ['pool'] + ['conv_36']*3 + ['pool'] + ['FC_64'] # VGGlite
    # struct = ['conv_16']*2 + ['pool'] + ['conv_32']*3 + ['pool'] + ['conv_64']*3 + ['pool'] + ['FC_128']*2 # VGGlite2

    vgg = VGGTest(struct)
    num_samples_ratio = 0.3 # 训练数据百分比

    vgg.load_train_data(num_samples_ratio)

    epoch_more = 4
    lr = 10**-3
    reg = 10**-4
    batch = 64 
    lr_decay = 0.8 
    mu = 0.9 
    optimizer = 'nesterov'
    regularization = "L2"
    activation = "ReLU"

    vgg.train(epoch_more ,  lr , reg , batch, lr_decay , mu , 
                optimizer = 'nesterov', regularization = "L2", activation = "ReLU")

    print('-----------------------------------------end')
    