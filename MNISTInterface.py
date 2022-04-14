# 2022 3.16 
# yimingli

from random import shuffle
import numpy as np
import gzip, struct

'''
MNIST 数据导入类
'''
class MNISTInterface():

    def load_train_data(self, num_ratio):
        '''
        载入训练集，随机分为训练集和验证集
        '''
        print('-------------------------------------load train data')
        # 
        imgs, labels = self.get_mnist_train()
        imgs = imgs / 255
        # 
        self.num_samples = labels.size

        if isinstance(num_ratio, int):
            self.num_train_samples = num_ratio
        else:
            self.num_train_samples = int( self.num_samples * num_ratio )

        self.num_val_samples = self.num_samples - self.num_train_samples
        shuffle_no = list(range(self.num_samples))
        np.random.shuffle(shuffle_no)
        imgs = imgs[shuffle_no]
        labels = labels[shuffle_no]

        self.train_data = imgs[0:self.num_train_samples]
        self.train_labels = labels[0:self.num_train_samples]
        self.val_data = imgs[self.num_train_samples::]
        self.val_labels = labels[self.num_train_samples::]
        self.set_data_pro()
        
    def load_test_data(self):
        '''
        载入测试集
        '''
        print('-------------------------------------load test data')
        imgs, labels = self.get_mnist_test()
        imgs = imgs/255
        self.test_data = imgs
        self.test_labels = labels
        # 
        self.set_data_pro()
        
    def set_data_pro(self, num_class= 10, im_height= 28, im_width= 28, im_dims= 1):
        '''
        数据集超参数设置
        '''
        self.num_class = num_class
        self.im_height = im_height
        self.im_width = im_width
        self.im_dims = im_dims

    def _read(self, image, label):
        mnist_dir = 'D:\\ScientificResearch\\MachineLearning\\ConvolutionalNeuralNetworks\\MNIST\\'
        with gzip.open(mnist_dir +label) as flbl:
            magic, num = struct.unpack(">II", flbl.read(8))
            label = np.fromstring(flbl.read(), dtype= np.uint8)

        with gzip.open(mnist_dir + image, 'rb') as fimg:
            magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
            image = np.fromstring(fimg.read(), dtype= np.uint8).reshape(len(label), rows, cols)
            
        return image, label

    def get_mnist_train(self):
        train_img, train_label = self._read('train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz')
        train_img = train_img.reshape((*train_img.shape,1))
        return train_img, train_label

    def get_mnist_test(self):
        test_img, test_label = self._read('t10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz')
        test_img = test_img.reshape((*test_img.shape,1))
        return test_img, test_label 
