# 2022 3.16 - 3.26
# yimingli 


import struct

from VGG import VGG

from CNNtrainInterface import CNNtarinInterface

from MNISTInterface import MNISTInterface

class VGGTest(VGG,  CNNtarinInterface,  MNISTInterface):
    pass

if __name__ == '__main__':

    print('-------------------------------------start')
    # struct = [] # 线性模型
    # struct = ['FC_64'] # 只含一个隐含层
    struct = ['conv_8'] + ['pool'] + ['conv_12']*3 + ['pool'] + ['conv_36']*3 + ['pool'] + ['FC_64']

    vgg = VGGTest(struct)
    num_samples = 0.7

    vgg.load_train_data(num_samples)

    
    train = True
    search_train = False
    if train:
        if search_train:
            print('---------------------------------------------------------train random search')
            vgg.train_random_search(lr=[-1.0,-5.0], reg = [-3,-5], num_try= 10, epoch_more = 2, 
            batch = 64, lr_decay = 1, mu = 0.9, optimizer = 'nesterov', regularization = 'L2', activation = 'ReLU')
        else:
            vgg.train_from_checkpoint(epoch_more = 2, checkpoint_f_name= 'checkpoint_(loss_-0.98)_(epoch2)__[(lr reg)_(-3.29 -4.0)]_ nesterov ReLU L2.npy')
    else:
        vgg.test_from_checkpoint( 'checkpoint_(loss_-0.8)_(epoch2)__[(lr reg)_(-3.19 -4.0)]_ nesterov ReLU L2.npy')
    
    print('-----------------------------------------end')

    '''
    vgg.featuremap_shape()
    vgg.init_params()
    epoch_more = 2
    lr = 10**-3
    reg = 10**-4
    batch = 64 
    lr_decay = 0.8 
    mu = 0.9 
    optimizer = 'nesterov'
    regularization = "L2"
    activation = "ReLU"
    vgg.context = [lr, reg, batch, lr_decay, mu, optimizer, regularization, activation]
    vgg.train(epoch_more = 2,  lr = 10**-3,  reg = 10**-4,batch = 64, lr_decay = 0.8, mu = 0.9, 
                optimizer = 'nesterov', regularization = "L2", activation = "ReLU")

    print('-----------------------------------------end')
    '''

