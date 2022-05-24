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
    '''
    last FC 不用定义在 strcut 中
    '''
    # struct = [] # 线性模型
    # struct = ['FC_64'] # 只含一个隐含层
    # struct = [conv_6_5_2_2] + [pool] + [] + [pool] + [] + [] + [] # LetNEet
    # struct = ['conv_8'] + ['pool'] + ['conv_12']*3 + ['pool'] + ['conv_36']*3 + ['pool'] + ['FC_64'] # VGGlite
    struct = ['conv_16']*2 + ['pool'] + ['conv_32']*3 + ['pool'] + ['conv_64']*3 + ['pool'] + ['FC_128']*2 # VGGlite2

    vgg = VGGTest(struct)
    num_samples_ratio = 0.7

    vgg.load_train_data(num_samples_ratio)

    train = True
    search_train = False
    if train:
        if search_train:
            print('---------------------------------------------------------train random search')
            vgg.train_random_search(lr=[-1.0,-5.0], reg = [-3,-5], num_try= 10, epoch_more = 2, 
                    batch = 64, lr_decay = 1, mu = 0.9, optimizer = 'nesterov', regularization = 'L2', activation = 'ReLU')
        else:
            vgg.train_from_checkpoint(epoch_more = 2, checkpoint_f_name= 'checkpoint_(accuracys_0.9849)_(loss_-0.96)_(epoch_retrain2)_(lr_-3.29)_(reg_-4.0)_nesterov_ReLU_L2.npy')
    else:
        vgg.test_from_checkpoint( 'checkpoint_(loss_-0.8)_(epoch2)__[(lr reg)_(-3.19 -4.0)]_ nesterov ReLU L2.npy')
    
    print('-----------------------------------------end')

