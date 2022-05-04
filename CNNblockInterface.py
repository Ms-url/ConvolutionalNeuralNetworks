# 2022.3.4 - 3.6
# YiMimgLi

import numpy as np
from ActivationInterface import ActivationInterface

'''
卷积网络基本模块类
    卷积层、池化层、全连接层、softmax损失函数 的向前和反向计算
    参数初始化
#####################################################
conv_layer                  卷积层批量计算
d_conv_layer
pool_layer                  池化层批量计算
d_pool_layer 
fully_connected_layer       全连接层批量计算
d_fully_connected_layer 
softmax_layer               softmax层计算

data_loss                   计算数据损失
d_scores                    获取分值梯度
param_init                  权重参数初始化

#####################################################
'''

class CNNblockInterface(ActivationInterface):

    @staticmethod
    def conv_layer(in_data, weights, biases, layer_param = (0,3,1,1) ,activation = "ReLU"):
        '''
        卷积层批量计算
            indata.shape = [batch, in_height, in_width, in_depth]
            weigths.shape = [filter_size ^ 2 * in_height, out_depth]
            biases.shape = [1, out_depth]
            out_data = [batch, out_height, out_width, out_depth]
        计算梯度所需数据
            matric_data, filter_data
        '''
        # weights 2D卷积核组
        # layer_param 卷积核超参数 （尺寸，步长，填充数）

        filter_size = layer_param[1]
        stride = layer_param[2]
        padding = layer_param[3]

        [batch, in_height, in_width, in_depth] = in_data.shape  
        out_depth = biases.shape[1]

        out_height = (in_height - filter_size + 2*padding )//stride + 1
        out_width = (in_width - filter_size + 2*padding )//stride + 1
        out_size = out_height * out_width
        out_data = np.zeros( (batch, out_height, out_width, out_depth) )

        # 2D大矩阵储存空间
        # 输出矩阵的每个点都由卷积运算得出，因此行数为 out_size 乘以处理的特征图批量
        # 列为 2D卷积核乘以 输入深度
        matric_data = np.zeros((out_size*batch ,in_depth*filter_size**2))

        padding_data = np.zeros( (batch , in_height + 2*padding , in_width + 2*padding ,in_depth ) )
        if padding == 0:
            padding_data[:,:,:,:] = in_data
        else:
            padding_data[:,padding:-padding,padding:-padding,:] = in_data

        height_f = padding_data.shape[1] - filter_size + 1 
        width_f = padding_data.shape[2] - filter_size + 1

        # TODT
        for i_batch in range(batch):
            i_batch_size = i_batch * out_size

            for i_h , i_height in zip(range(out_height), range(0,height_f,stride)):
                i_height_size = i_batch_size + i_h * out_width

                for i_w , i_width in zip(range(out_width) , range(0,width_f,stride)):
                    matric_data[ i_height_size + i_w , : ] =      \
                    padding_data[ i_batch , i_height : i_height + filter_size , i_width : i_width + filter_size, : ].ravel()

        # 卷积运算 + 激活
        after_filtering_data = np.dot(matric_data , weights) + biases
        filter_data = ActivationInterface.activation(after_filtering_data,activation)

        # filter_data 转化成特征图
        for i_batch in range(batch):
            i_batch_size = i_batch * out_size

            for i_height in range(out_height):
                i_height_size = i_batch_size + i_height * out_width

                for i_width in range(out_width):
                    out_data[i_batch , i_height, i_width , : ] = filter_data[i_height_size + i_width , : ]

        return out_data, matric_data, filter_data
    
    @staticmethod
    def d_conv_layer(d_out_data, matric_data, filter_data, weights, maps_shape, layer_param = (3,1,1) , activation = "ReLU" ):
        '''
        卷积层梯度反向传播
        由输出梯度反向计算输入梯度
        d_out_data, matric_data, filter_data, weights 在网络向前计算中产生
            maps_shape 保存 in_height,in_width,in_depth
        '''
        # 1.输出特征图的梯度变换为矩阵
        # 2.全连接层和激活层梯度反向传递
        # 3.大矩阵变换为特征图型状梯度

        filter_size = layer_param[0]
        stride = layer_param[1]
        padding = layer_param[2]

        in_height,in_width,in_depth = maps_shape
        [batch, out_height, out_width, out_depth] = d_out_data.shape
        out_size = out_height * out_width
        
        # d_out_data 变换为 d_filter_data
        # d_out_data 每个深度列为 d_filter_data 的一行
        d_filter_data = np.zeros_like( filter_data )
        for i_batch in range(batch):
            i_batch_size = i_batch * out_size

            for i_height in range(out_height):
                i_height_size = i_batch_size + i_height * out_width

                for i_width in range(out_width):
                    d_filter_data[i_height_size + i_width, :] = d_out_data[i_batch, i_height, i_width, :]
        
        # 激活层反向传递
        d_filter_data = ActivationInterface.d_activation( d_filter_data , filter_data, activation )
        # 全连接层反向传递
        d_weights = np.dot(matric_data.T, d_filter_data)
        d_biases = np.sum(d_filter_data, axis=0 , keepdims=True)
        d_matric_data = np.dot(d_filter_data, weights.T)

        padding_height = in_height + 2*padding
        padding_width = in_width + 2*padding
        d_padding_data = np.zeros( (batch, padding_height, padding_width, in_depth) )

        height_f = padding_height - filter_size + 1 
        width_f = padding_width - filter_size + 1

        for i_batch in range(batch):
            i_batch_size = i_batch * out_size

            for i_h , i_height in zip(range(out_height), range(0,height_f,stride)):
                i_height_size = i_batch_size + i_h * out_width

                for i_w , i_width in zip(range(out_width) , range(0,width_f,stride)):
                    d_padding_data[ i_batch , i_height : i_height + filter_size , i_width : i_width + filter_size, : ] +=    \
                    d_matric_data[ i_height_size + i_w , : ].reshape(filter_size, filter_size, -1)
        
        if padding:
            d_in_data = d_padding_data[:, padding:-padding, padding:-padding, :]
        else :
            d_in_data = d_padding_data

        return d_weights, d_biases, d_in_data

    @staticmethod
    def pool_layer(in_data, filter_size = 2,stride = 2):
        '''
        池化层计算
        '''
        [batch,in_height,in_width,in_depth] = in_data.shape
        out_height = (in_height- filter_size)//stride + 1
        out_width = (in_width- filter_size)//stride + 1
        out_depth = in_depth
        out_size = out_height * out_width 

        out_data = np.zeros( (batch, out_height, out_width, out_depth) )
        # 每一个输出数据都对应 out_depth 行深度特征
        # 每行为4个局部窗口元素
        # n x n x 96 的数据进行 2 x 2 池化，每个 2 x 2 x 96 的局部窗口元素需拉伸为 96 x 4 的矩阵
        matric_data = np.zeros( (batch* out_size* out_depth, filter_size**2))

        height_f = in_height - filter_size + 1 
        width_f = in_width - filter_size + 1

        for i_batch in range(batch):
            i_batch_size = out_size * i_batch * in_depth

            for i_h , i_height in zip(range(out_height), range(0,height_f,stride)):
                i_height_size = i_batch_size + i_h * out_width * in_depth

                for i_w , i_width in zip(range(0, out_width* in_depth, in_depth ) , range(0,width_f,stride)):
                    # md 按引用调用
                    md = matric_data[ i_height_size + i_w :i_height_size + i_w +in_depth, :]
                    src = in_data[i_batch, i_height: i_height+filter_size, i_width: i_width+ filter_size ,:]

                    for i in range(filter_size):
                        for j in range(filter_size):
                            # 深度特征拉伸为列向量储存
                            md[:, i*filter_size + j] = src[i,j,:]

        # 取行最大值进行池化
        matric_data_max_value = matric_data.max(axis=1, keepdims=True)
        matric_data_max_pos = (matric_data == matric_data_max_value)

        for i_batch in range(batch):
            i_batch_size = i_batch * out_size * out_depth

            for i_height in range(out_height):
                i_height_size = i_batch_size + i_height * out_width * out_depth

                for i_width in range(out_width):
                    out_data[ i_batch, i_height, i_width, :] =     \
                    matric_data_max_value[i_height_size + i_width* out_depth :i_height_size + i_width* out_depth + out_depth].ravel() 
 
        return out_data , matric_data_max_pos
        
    @staticmethod
    def d_pool_layer(d_out_data, matric_data_max_pos, maps_shape,filter_size = 2,stride = 2):
        '''
        池化层梯度反向传播
            d_out_data 
                上一次梯度反向传播得到
            matric_data_max_pos 
                记录了输入特征图中最大值的位置,尺寸为变换后的大矩阵,每行4个局部窗口元素
        '''
        in_height, in_width, in_depth = maps_shape
        batch, out_height, out_width, out_depth = d_out_data.shape
        out_size = out_height * out_width

        # 最大值元素处的输入梯度等于输出梯度，其他位置梯度为 0
        matric_data_not_max_pos = ~ matric_data_max_pos

        d_in_data = np.zeros( (batch, in_height, in_width, in_depth ),dtype=np.float64 )

        height_f = in_height - filter_size + 1 
        width_f = in_width - filter_size + 1

        for i_batch in range(batch):
            i_batch_size = out_size * i_batch * in_depth

            for i_h , i_height in zip(range(out_height), range(0, height_f, stride)):
                i_height_size = i_batch_size + i_h * out_width * out_depth

                for i_w_dout, i_w , i_width in zip(range(out_width),range(0, out_width* in_depth, in_depth ) , range(0, width_f, stride)):
                    # md、d_in、d_out 按引用调用
                    md = matric_data_not_max_pos[i_height_size+ i_w: i_height_size + i_w + in_depth, :]
                    d_in = d_in_data[i_batch, i_height: i_height+ filter_size, i_width: i_width+ filter_size, :]
                    d_out = d_out_data[i_batch, i_h, i_w_dout, :]

                    for i in range(filter_size):
                        for j in range(filter_size):
                            d_in[i,j,:] = d_out[:] 
                            ############################# TDOT
                            d_in[i,j,:][md[:, i*filter_size + j]] = 0 

        return d_in_data

    @staticmethod
    def fully_connected_layer(in_data,weights,biases,out_depth, last=False ,activation = "ReLU"):
        '''
        全连接层
            indata.shape = [batch, in_height, in_width, in_depth]
            weigths.shape = [ in_height*in_width*in_depth , out_depth]
            biases.shape = [1, out_depth]
            out_data = [batch, out_height, out_width, out_depth]
        计算梯度所需数据
            matric_data, filter_data
        '''
        # 全连接层与常规神经网络相同
        [batch, in_height, in_width, in_depth] = in_data.shape  
        matric_data = np.zeros( (batch , in_height*in_width*in_depth ) )

        for i_batch in range(batch):
            matric_data[i_batch] = in_data[i_batch].ravel()

        filter_data = np.dot(matric_data, weights) + biases

        # 激活
        if not last:
            filter_data = ActivationInterface.activation(filter_data, activation)

        # 使用4D 矩阵储存输出数据
        out_data = np.zeros((batch,1,1,out_depth))
        for i_batch in range(batch):
            out_data[i_batch] = filter_data[i_batch]

        return out_data, matric_data , filter_data

    @staticmethod
    def d_fully_connected_layer(d_out_data, matric_data, filter_data, weights, maps_shape, last=False, activation = "ReLU"):
        '''
        d_out_data, matric_data, filter_data 在网络向前计算中产生

        '''
        in_height, in_width, in_depth = maps_shape
        batch = d_out_data.shape[0] 

        d_filter_data = np.zeros_like(filter_data)
        for i_batch in range(batch):
            d_filter_data[i_batch] = d_out_data[i_batch].ravel()
        
        # 激活函数梯度反向传播
        if not last:
            d_filter_data = ActivationInterface.d_activation(d_filter_data, filter_data, activation)
        
        # TODT
        d_weights = np.dot(matric_data.T, d_filter_data)
        d_biases = np.sum(d_filter_data, axis=0, keepdims=True)
        d_matric_data = np.dot(d_filter_data, weights.T)

        d_in_data = np.zeros( (batch, in_height, in_width, in_depth) )

        for i_batch in range(batch):
            d_in_data[i_batch] = d_matric_data[i_batch].reshape(in_height, in_width, -1) 

        return d_weights, d_biases, d_in_data

    @staticmethod
    def softmax_layer(scores):
        '''
        scores.shape = [batch,1,1,in_depth]
        probs.shape = [batch,1,1,in_depth]
        '''
        # softmax 损失定义
        # Li = -log(e^si/sum(s))

        scores -= np.max(scores,axis=3,keepdims=True)
        exp_scores = np.exp(scores) + 10**(-6)
        exp_scores_sum = np.sum(exp_scores, axis=3, keepdims=True)
        probs = exp_scores / exp_scores_sum
        # -log(probs)
        return probs

    @staticmethod
    def data_loss(probs,labels):
        '''
        probs.shape = [batch,1,1,in_depth]
        labels 标签
        '''
        probs_correct = probs[range(probs.shape[0]),:,:,labels]
        logprobs_correct = -np.log(probs_correct)
        # 归一化
        data_loss = np.sum(logprobs_correct) / labels.shape[0]
        return data_loss

    @staticmethod
    def d_scores(probs, labels):
        '''
        获取分值梯度
            probs.shape = [batch,1,1,in_depth]
            labels 标签
        '''
        dscores = probs.copy()
        dscores[range(probs.shape[0]),:,:,labels] -= 1
        dscores /= labels.shape[0]
        return dscores

    @staticmethod
    def param_init(out_depth,in_depth,filter_size2):
        '''
        权重参数初始化
            weights.shape = [in_depth*filter_size**2, out_depth]
            filter_size 滤波器展开成一维的尺寸，奇数
        '''
        std = np.sqrt(2/(in_depth*filter_size2))
        weights = std * np.random.randn(in_depth*filter_size2,out_depth)
        biases = np.zeros((1,out_depth))
        return weights, biases



