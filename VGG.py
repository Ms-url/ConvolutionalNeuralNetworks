# 2022 3.10 - 3.13
# yimingli

import numpy as np
import re
from CNNblockInterface import CNNblockInterface

'''
VGG 类
    VGG 网络结构
###################################################
__init__                输入struct
struct_parse            对struct 进行解析
featuremap_shape        由struct 计算各层特征图尺寸
init_params             各层网络权重和偏置初始化    -> CNNblockterface -> init_params

reg_loss                计算正则化损失
d_weight_reg            正则化梯度

forward                 向前传播过程        -> CNNblockterface
backpropagation         反向传播过程        -> CNNblockterface
params_updata           参数更新过程

save_checkpoint         保存模型
load_checkpoint         载入保存的模型

predict                 预测准确率
###################################################
'''

class VGG(CNNblockInterface):
    '''
    last FC 不用定义在 strcut 中
    '''
    
    def __init__(self, struct = []) -> None:
        if len(struct)==0:
            print("A linear model is being used")
        self.struct_parse(struct)
        self.struct = struct
        self.struct += ['FC'] 
        
    def struct_parse(self, struct):
        '''
        对struct 进行解析
        '''
        layers = []
        for layer in struct:
            convfull = re.match('^conv_(\d{1,3})_(\d{1})_(\d{1})_(\d{1})$', layer)
            convdefault = re.match('^conv_(\d{1,3})$', layer)
            pool = re.match('^pool$',layer)
            fc = re.match('^FC_(\d{1,4})$', layer)

            if convfull:
                layers.append( (int(convfull.group(1)), int(convfull.group(2)), int(convfull.group(3)), int(convfull.group(4)), 'conv' ) )
            elif convdefault:
                layers.append( ( int(convdefault.group(1)),3,1,1,'conv') )
            elif pool:
                layers.append( ( layers[-1][0] , 'pool' ) )
            elif fc:
                layers.append( ( int(fc.group(1)), 'FC' ) )
            else:
                raise ValueError(""" 只能使用 : conv_depth_size_stride_stride 或 conv_depth 或 pool 或 FC_depth """)

        layers.append(('','Last_FC'))
        self.layers_params = layers

    def featuremap_shape(self):
        '''
        计算各层特征图尺寸
        '''
        maps_shape = []
        in_map_shape = (self.im_height, self.im_width, self.im_dims)
        maps_shape.append( in_map_shape )

        for layer in self.layers_params:
            if layer[-1] == 'Last_FC':
                break
            elif layer[-1] == 'FC':
                in_map_shape = (1,1,layer[0])
            elif layer[-1] == 'conv':
                (out_depth, filter_size, stride, padding, not_used) = layer
                
                out_height = (in_map_shape[0] - filter_size + 2*padding)//stride + 1
                out_width = (in_map_shape[1] - filter_size + 2*padding)//stride + 1
                in_map_shape = (out_height, out_width, out_depth)

                if out_height < filter_size or out_width < filter_size:
                    raise ValueError("""not compatible with the image size""")

            elif layer[-1] == 'pool':
                filter_size = 2
                stride = 2

                out_height = (in_map_shape[0] - filter_size)//stride + 1
                out_width = (in_map_shape[1] - filter_size)//stride + 1
                in_map_shape = (out_height, out_width, layer[0])

                if out_height < filter_size or out_width < filter_size:
                    raise ValueError("""not compatible with the image size\n""")
            else:
                pass

            maps_shape.append(in_map_shape)

        self.maps_shape = maps_shape

    def init_params(self):
        '''
        网络权重和偏置初始化
        '''
        self.weights = []
        self.biases = []
        # 
        in_depth = self.im_dims
        out_depth = in_depth

        for layer_param, map_shape in zip(self.layers_params, self.maps_shape):
            weight = np.array([])
            bias = np.array([])

            # 
            # print(f'param init layer param----{layer_param[-1]}')

            if layer_param[-1] == 'Last_FC':
                in_depth = out_depth
                out_depth = self.num_class
                # CNN block param_init()
                (weight, bias) = self.param_init(out_depth, in_depth, map_shape[0]*map_shape[1])

            elif layer_param[-1] == 'FC':
                out_depth = layer_param[0]
                in_depth = map_shape[2]
                # CNN block param_init()
                (weight, bias) = self.param_init(out_depth, in_depth, map_shape[0]*map_shape[1])

            elif layer_param[-1] == 'conv':
                filter_size = layer_param[1]
                out_depth = layer_param[0]
                # CNN block param_init()
                (weight, bias) = self.param_init(out_depth, in_depth, filter_size*filter_size)

            elif layer_param[-1] == 'pool':
                pass
            else:
                pass
            in_depth = out_depth
            
            self.weights.append(weight)
            self.biases.append(bias)

        # TDOT
        self.v_weights = []
        self.v_biases = []
        self.cache_biases = []
        self.cache_weights = [] 
        for weight, bias in zip(self.weights, self.biases):
            self.v_weights.append(np.zeros_like(weight))
            self.v_biases.append(np.zeros_like(bias))
            self.cache_weights.append(np.zeros_like(weight))
            self.cache_biases.append(np.zeros_like(bias))

    def reg_loss(self, reg = 10**-5, regularization = 'L2'):
        '''
        计算正则化损失
        '''
        reg_loss = 0
        # self.weights VGG init_params()中生成
        for weight in self.weights:
            if weight.size !=0:
                # RegularizationInterface
                reg_loss += self.norm_regularization(weight, regularization, reg)
                 

        return reg_loss

    def forward(self, batch_data, labels, reg= 10**-5, regularization= 'L2', activation = 'ReLU'):
        '''
        向前计算
        '''
        # TDOT
        self.matric_data = []
        self.filter_data = []
        self.matric_data_max_pos = []

        in_maps = batch_data
        for layer_param, weight, bias in zip(self.layers_params, self.weights, self.biases):
            matric_data = np.array([])
            filter_data = np.array([])
            matric_data_max_pos = np.array([])

            # 
            # print(f'forward-----{layer_param[-1]}')

            if layer_param[-1] == 'Last_FC':
                # CNN block fully_connected_layer()
                ( out_maps, matric_data, filter_data) = self.fully_connected_layer(in_maps, weight, bias, self.num_class, True , activation) 
                # 最后一层不需激活
            elif layer_param[-1] == 'FC':
                # CNN block fully_connected_layer()
                (out_maps, matric_data, filter_data) = self.fully_connected_layer(in_maps, weight, bias, layer_param[0], False , activation)
            elif layer_param[-1] == 'conv':
                # CNN block conv_layer()
                (out_maps, matric_data, filter_data) = self.conv_layer(in_maps, weight, bias, layer_param[0:-1], activation)
            elif layer_param[-1] == 'pool':
                # CNN block pool_layer()
                (out_maps, matric_data_max_pos) = self.pool_layer(in_maps ) 
                # print(out_maps.shape)
            else:
                pass
            in_maps = out_maps
            # 
            # print(f'forward-----{layer_param[-1]}{(out_maps.shape)}')

            self.matric_data.append(matric_data)
            self.filter_data.append(filter_data)
            self.matric_data_max_pos.append(matric_data_max_pos)

        print('---------------------------forward end')

        # CNN block softmax_layer()
        self.probs = self.softmax_layer(out_maps) # self.probs
        # CNN block data_loss()
        data_loss = self.data_loss(self.probs, labels)
        # CNN VGG reg_loss()
        reg_loss = self.reg_loss(reg, regularization)

        print(f"data loss = {data_loss}")
        print(f"reg loss = {reg_loss}")

        return data_loss, reg_loss

    def predict(self, batch_data, labels, activation = 'ReLU'):
        '''
        预测准确率
        '''
        in_maps = batch_data

        for layer_param, weight, bias in zip(self.layers_params, self.weights, self.biases):
            if layer_param[-1] == 'Last_FC':
                # CNN block fully_connected_layer()
                (out_maps, matric_data, filter_data) = self.fully_connected_layer(in_maps, weight, bias, self.num_class, True , activation) 
                # 最后一层不需激活
            elif layer_param[-1] == 'FC':
                # CNN block fully_connected_layer()
                (out_maps, matric_data, filter_data) = self.fully_connected_layer(in_maps, weight, bias, layer_param[0], False , activation)
            elif layer_param[-1] == 'conv':
                # CNN block conv_layer()
                (out_maps, matric_data, filter_data) = self.conv_layer(in_maps, weight, bias, layer_param[0:-1], activation)
            elif layer_param[-1] == 'pool':
                # CNN block pool_layer()
                (out_maps, matric_data_max_pos) = self.pool_layer(in_maps ) 
            else:
                pass
            in_maps = out_maps

        predicated_class = np.argmax(out_maps, axis= 3)
        # print(predicated_class)
        accuracy = (predicated_class.ravel() == labels)
        # print(accuracy)
        return np.mean(accuracy)

    def d_weight_reg(self, reg= 10**-5, regularization= 'L2'):
        '''
        正则化梯度
        '''
        # TDOT
        for i in range(len(self.weights)):
            weight = self.weights[i]
            if weight.size != 0:
                # RegularizationInterface
                self.d_weights[-1-i] += self.d_norm_regularization(weight, regularization, reg)
   
    def backpropagation(self, labels, reg= 10**-5, regularization= 'L2', activation = 'ReLU'):
        '''
        反向传播
        '''
        # CNN block d_scores()
        d_scores = self.d_scores(self.probs, labels)
        d_out_maps = d_scores

        self.d_weights = []
        self.d_biases = []
        for layer_param, maps_shape, weight, matric_data, filter_data, matric_data_max_pos in  \
            zip(reversed(self.layers_params), reversed(self.maps_shape), reversed(self.weights), reversed(self.matric_data), \
            reversed(self.filter_data), reversed(self.matric_data_max_pos)):
            
            # 
            # print(f'back propagation---{layer_param[-1]}')

            if layer_param[-1] == 'Last_FC':
                # CNN block d_fully_connected_layer()
                d_weight, d_bias, d_in_maps = self.d_fully_connected_layer(d_out_maps, matric_data, filter_data, weight, maps_shape, True, activation)
            elif layer_param[-1] == 'FC':
                # CNN block d_fully_connected_layer()
                d_weight, d_bias, d_in_maps = self.d_fully_connected_layer(d_out_maps, matric_data, filter_data, weight, maps_shape, False, activation)
            elif layer_param[-1] == 'conv':
                # CNN block d_conv_layer()
                d_weight, d_bias, d_in_maps = self.d_conv_layer(d_out_maps, matric_data, filter_data, weight, maps_shape, layer_param[1:-1], activation )
            elif layer_param[-1] == 'pool':
                # CNN block d_pool_layer()
                d_weight = np.array([])
                d_bias = np.array([])
                d_in_maps = self.d_pool_layer(d_out_maps, matric_data_max_pos, maps_shape, filter_size = 2,stride = 2)
            else:
                pass
        
            d_out_maps = d_in_maps
            self.d_weights.append(d_weight)
            self.d_biases.append(d_bias)
            self.d_batch_data = d_in_maps
        # VGG d_weight_reg()
        self.d_weight_reg(reg, regularization)
            
        print(f'---------------------------back propagation end')

    def params_updata(self, lr= 10**-4, t= 0, mu= 0.9, optimizer = 'nesterov' ):
        '''
        参数更新
        '''
        self.updata_ratio = []
        for i in range(len(self.weights)):
            # 
            weight = self.weights[i]
            bias = self.biases[i]
            # 
            d_weight = self.d_weights[-1-i]
            d_bias = self.d_biases[-1-i]
            # 
            v_weight = self.v_weights[i]
            v_bias = self.v_biases[i]
            # 
            cache_weights = self.cache_weights[i]
            cache_biases = self.cache_biases[i]

            if weight.size != 0:
                # GradientOptimizerInterface
                update_ratio_w, weight, v_weight = self.optimizerInterface(optimizer,lr, weight, d_weight, v_weight, mu, cache_weights, t)
                update_ratio_b, bias, v_bias = self.optimizerInterface(optimizer,lr, bias, d_bias, v_bias, mu , cache_biases, t)

                self.weights[i] = weight
                self.biases[i] = bias
                
                self.v_weights[i] = v_weight
                self.v_biases[i] =v_bias

                self.cache_weights[i] = cache_weights
                self.cache_biases[i] = cache_biases

                self.updata_ratio.append((update_ratio_w, update_ratio_b))

    def save_checkpoint(self, f_name):
        '''
        保存数据
        '''
        with open(f_name, 'wb') as f:
            np.save(f, np.array([3,1,4,1,5,9,2,8,8])) # 魔术数
            np.save(f, np.array(self.struct))
            np.save(f, np.array( [self.num_class, self.im_dims, self.im_height, self.im_width]))
            np.save(f, np.array( self.layers_params ) )
            np.save(f, np.array( self.maps_shape ))
            np.save(f, np.array( self.context ))
            for array in self.weights:
                np.save(f, array)
            for array in self.biases:
                np.save(f, array)
            for array in self.v_weights:
                np.save(f, array)
            for array in self.v_biases:
                np.save(f, array)
            for array in self.cache_weights:
                np.save(f, array)
            for array in self.cache_biases:
                np.save(f, array)

    def load_checkpoint(self, f_name):
        '''
        载入保存的模型
        '''
        with open(f_name, 'rb') as f:
            magic_munber = np.load(f,allow_pickle=True)
            if not all(magic_munber == np.array([3,1,4,1,5,9,2,8,8])):
                raise ValueError('file format wrong')
            self.struct = np.load(f)
            print('\n The net struct is:',self.struct)
            im_property = np.load(f)
            self.num_class, self.im_dims, self.im_height, self.im_width = im_property
            self.layers_params = np.load(f,allow_pickle=True)
            self.maps_shape = np.load(f)
            self.context = np.load(f)

            self.weights = []
            self.biases = []
            for i in range(len(self.layers_params)):
                array = np.load(f)
                self.weights.append(array)

            for i in range(len(self.layers_params)):
                array = np.load(f,allow_pickle=True)
                self.biases.append(array)

            self.v_weights = []
            self.v_biases = []
            for i in range(len(self.layers_params)):
                array = np.load(f)
                self.v_weights.append(array)

            for i in range(len(self.layers_params)):
                array = np.load(f)
                self.v_biases.append(array)

            self.cache_biases = []
            self.cache_weights = []
            
            for i in range(len(self.layers_params)):
                array = np.load(f)
                self.cache_weights.append(array)

            for i in range(len(self.layers_params)):
                array = np.load(f)
                self.cache_biases.append(array)

            print( 'struct:', self.layers_params )


