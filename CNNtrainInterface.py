# 2022 3.7 - 3.16
# YiMingLi

import matplotlib.pyplot as plt 
import numpy as np
from ActivationInterface import ActivationInterface
from GradientOptimizaerInterface import GradientOptimizerInterface
from RegularizationInterface import RegularizationInterface

'''
卷积网络训练类
    封装了训练过程
########################################
shuffle_data                数据打乱
methods_check               方式检测

gen_lr_reg                  获取随机学习率和正则化系数
train                       训练网络                   -> VGG*(模型类) -> all
train_random_search         使用超参数搜索训练网络      -> train  &  gen_lr_reg  &  methods_check

test                        测试模型
train_from_checkpoint       加载已有模型训练            -> test             
test_from_checkpoint        测试已有模型                -> test
########################################
'''
class CNNtarinInterface(ActivationInterface, GradientOptimizerInterface, RegularizationInterface):
    '''
    self.context = [lr, reg, batch, lr_decay, mu, optimizer, regulation, activation]
    
    '''
    def shuffle_data(self):
        '''
        数据打乱
        '''
        shuffle_no = list(range(self.num_train_samples))
        np.random.shuffle(shuffle_no)
        self.train_labels = self.train_labels[shuffle_no]
        self.train_data = self.train_data[shuffle_no]

        shuffle_no = list(range(self.num_val_samples))
        np.random.shuffle(shuffle_no)
        self.val_labels = self.val_labels[shuffle_no]
        self.val_data = self.val_data[shuffle_no]

    def methods_check(self, activation, regularization, optimizer):
        '''
        方法检查
            optimizer 为函数指针
        '''
        self.check_activation(activation)
        self.check_norm_regularization(regularization)
        self.check_optimizer(optimizer)

    def train_random_search(self, lr = [-1, -5], reg = [-1, -5], num_try = 10, epoch_more = 1, batch = 64, 
                            lr_decay = 0.8, mu = 0.9, optimizer = 'nesterov', regularization = 'L2', activation = 'ReLU'):
        '''
        随机搜索训练
        '''
        # CNN train methods_check()
        print('-------------------------------------methods check')
        self.methods_check(activation, regularization, optimizer )
        
        # 具体网络模型类接口 featuremap_shape() 计算各层特征图尺寸
        self.featuremap_shape()
        # CNN train gen_lr_reg
        lr_regs = self.gen_lr_reg(lr, reg, num_try)

        for lr_reg in lr_regs:
            try:
                print('----------------------------------------------train')
                self.train( epoch_more, *lr_reg, batch, lr_decay, mu, optimizer, regularization, activation, True )
            except KeyboardInterrupt:
                pass

    def train(self, epoch_more = 20,  lr = 10**-4,  reg = 10**-5,batch = 64, lr_decay = 0.8, mu = 0.9, 
                optimizer = 'nesterov', regularization = "L2", activation = "ReLU", search = False ):
        '''
        训练网络
        epoch 训练 代 次
        '''
        plt.close()
        fig = plt.figure('')
        ax = fig.add_subplot(3,1,1)
        ax.grid(True)
        plt.title(str(self.struct),fontsize=8)
        ax2 = fig.add_subplot(3,1,2)
        ax2.grid(True)
        plt.ylabel( 'update_ratio    accuracy    log(data loss) ' , fontsize = 14 )
        ax3 = fig.add_subplot(3,1,3)
        ax3.grid(True)
        plt.xlabel('log(lr) = '+ str(round( (np.log10(lr)),2 )) +'   '+ 'log(reg) = ' + str(round(np.log10(reg),2)) , fontsize = 12)

        # 具体网络模型类接口 featuremap_shape() 计算各层特征图尺寸
        if not search:
            self.featuremap_shape()
        # 具体网络模型类接口 init_params()
        self.init_params()
        # 
        # context
        # 
        self.context = [lr, reg, batch, lr_decay, mu, optimizer, regularization, activation]

        epoch = 0
        val_no = 0
        per_epoch_time = self.num_train_samples // batch

        while epoch < epoch_more:
            losses = 0
            print(f'---------------------------------------------------------------------------------epoch={epoch}')
            # CNN train shuffle_data
            self.shuffle_data()
            for i in range(0, self.num_train_samples, batch):
                print(f"#################################################### epoch {epoch}/{epoch_more}###### {i}/{self.num_train_samples}")
                batch_data = self.train_data[i:i+batch, : ]
                labels = self.train_labels[i:i+batch]

                # 具体网络模型类接口 forward()
                print('---------------------------forward')
                (data_loss, reg_loss) = self.forward(batch_data, labels, reg, regularization, activation)
                losses += data_loss + reg_loss
                # 具体网络模型类接口 backpropagation()
                print('---------------------------back propagation')
                self.backpropagation(labels, reg, regularization, activation)
                # 具体网络模型类接口 params_updata()
                print('---------------------------params_updata')
                self.params_updata( lr, per_epoch_time * epoch + i + 1 , mu, optimizer )
                # TDOT 具体网络模型类属性
                update_ratio = self.updata_ratio[0][0]

                if i % (batch * 20) == 0: 
                    ax.scatter( i/self.num_train_samples+ epoch, np.log10(data_loss), c= 'b', marker = '.')
                    # 具体网络模型类接口 predict()
                    # 训练集预测
                    print('---------------------------predict train_data')
                    train_accuracy = self.predict(batch_data, labels, activation)
                    batch_data_val = self.val_data[val_no: val_no + batch, : ]
                    labels_val = self.val_labels[val_no: val_no + batch]
                    # 具体网络模型类接口 predict()
                    # 验证集预测
                    print('---------------------------predict val_data')
                    val_accuracy = self.predict( batch_data_val, labels_val, activation )
                    val_no += batch

                    if val_no >= self.num_train_samples - batch:
                        val_no = 0
                    ax2.scatter( i/self.num_train_samples+ epoch, train_accuracy, c= 'r', marker = '*')
                    ax2.scatter( i/self.num_train_samples+ epoch, val_accuracy, c= 'b', marker = '.')
                    ax3.scatter( i/self.num_train_samples+ epoch, np.log10(update_ratio), c= 'r', marker = '.')
                    plt.pause(0.1)
                    
            epoch += 1
            self.context[0] = lr
            lr *= lr_decay

        plt.savefig('checkpoint_'+'(loss_'+str(round(np.log10(losses/per_epoch_time),2))+')_(epoch'+str(round(epoch,2))+')_'+
                        '_[(lr reg)_'+'('+str(round(np.log10(lr),2))+' '+str(round(np.log10(reg),2))+')]_'+
                        ' '+str(optimizer)+' '+str(activation)+' '+str(regularization) + '.png')
            
        print('--------------------------------save check point')
        self.save_checkpoint('checkpoint_'+'(loss_'+str(round(np.log10(losses/per_epoch_time),2))+')_(epoch'+str(round(epoch,2))+')_'+
                        '_[(lr reg)_'+'('+str(round(np.log10(lr),2))+' '+str(round(np.log10(reg),2))+')]_'+
                        ' '+str(optimizer)+' '+str(activation)+' '+str(regularization) + '.npy')

        self.test(batch, activation)        

    def gen_lr_reg(self, lr = [0, -6] ,reg = [-3, -6], num_try = 10 ):
        '''
        获取随机学习率和正则化系数
        '''
        minreg = min(reg)
        maxreg = max(reg)
        minlr = min(lr)
        maxlr = max(lr)
        randn = np.random.rand(num_try*2)
        lr_array = 10**(minlr + (maxlr - minlr)* randn[0: num_try])
        reg_array = 10**(minreg + (maxreg - minreg)*randn[num_try: num_try*2])
        lr_regs = zip(lr_array, reg_array)
        return lr_regs

    def train_from_checkpoint(self, epoch_more = 10, checkpoint_f_name = ''):
        '''
        加载已有模型训练
        '''
        # 具体网络模型类接口 load_checkpoint()
        self.load_checkpoint(checkpoint_f_name)
        [lr, reg, batch, lr_decay, mu, optimizer, regularization, activation] = self.context

        lr = np.double(lr)
        reg = np.double(reg)
        batch = np.int(batch)
        lr_decay = np.double(lr_decay)
        mu = np.double(mu)

        self.train(epoch_more , lr, reg ,batch , lr_decay , mu ,optimizer ,regularization, activation)

    def test_from_checkpoint(self,checkpoint_file_name):
        '''
        测试已有模型
        '''
        # 具体网络模型类接口 load_checkpoint()
        self.load_checkpoint(checkpoint_file_name)
        # context 
        [lr, reg, batch, lr_decay, mu, optimizer, regularization, activation] = self.context
        batch = np.int(batch)
        self.test(batch, activation)

    def test(self, batch, activation):
        '''
        测试模型
        '''
        # 数据集类接口 load_test_data() 
        self.load_test_data()
        # TDOT test_labels
        accuracys = np.zeros(shape=(self.test_labels.shape[0]))
        for i in range(0, self.test_labels.shape[0], batch):
            batch_data = self.test_data[i:i+batch, : ]
            label = self.test_labels[i:i+batch]
            accuracys[ i:i+batch ] = self.predict(batch_data, label, activation)

        accuracys = np.mean(accuracys)
        print( f'the test accuray: {accuracys}' )
        return accuracys
        

