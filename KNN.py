# 2022/4/23
# YiMingLi

from MNISTInterface import MNISTInterface
import numpy as np

class KNN(MNISTInterface):
    pass

if __name__ == '__main__':
    aim = KNN()
    k = int(input("输入k值："))

    num_samples_radio = 0.4 # 训练数据百分比
    # num_samples = 10000 # 训练数据数量

    aim.load_train_data(num_samples_radio)
    aim.load_test_data()

    print(aim.test_data.shape)
    batch,a,b,c =aim.train_data.shape

    cunt = 0
    right_num = 0
    for test,test_label in zip(aim.test_data,aim.test_labels):
        
        labels_num = np.zeros((batch,2))
        index = 0
        for train,train_label in zip(aim.train_data,aim.train_labels):
            diff = np.sum(np.abs(test-train))
            labels_num[index,0] = diff
            labels_num[index,1] = train_label
            index += 1
           
        data = labels_num[np.argsort(labels_num[:,0])]

        labels = np.zeros(10)
        for i in range(0,k):
            labels[int(data[k,1])] += 1

        poistion = np.argmax(labels)
    
        if poistion == test_label:
            right_num +=1
        
        cunt +=1
        # print(cunt)
        if cunt == 2000:
            break

    print(right_num/cunt)

       

        
    


