from batch_train import train
import glob
import random
import os
import gdal
import numpy as np
def load_smp_lst(td_pth,vd_pth,jdg=0):
    '''
    加载已有的训练数据和验证数据索引列表
    td_pth/vd_pth是训练和验证集的txt名字索引
    jdg为0的话就不会启动这个程序，算是一个保险吧。
    '''
    if jdg:
        with open(td_pth, 'r') as f:
            tra_data = f.readlines()
            print('成功加载已有训练集，包含样本{}个'.format(len(tra_data)))
        with open(vd_pth, 'r') as f:
            val_data = f.readlines()
            print('成功加载已有验证集，包含样本{}个'.format(len(val_data)))
        vd = []
        td = []
        for t in tra_data:
            td.append(t.strip('\n'))
        for v in val_data:
            vd.append(v.strip('\n'))
        return td,vd



fns = glob.glob(r'E:\SmokeDetection\source\MLP_Results\*')#不同框架的MLP的存储路径：命名形式:每层节点数_层数


#开始训练，根据文件夹上标注的层数和节点数训练
rootDataset = [r'I:\MLP_RESULT\20210209\256_3_100smoke1900nosmoke\training_pixels_100smoke1900nosmoke',
               r'I:\MLP_RESULT\20210209\256_3_200smoke1800nosmoke\training_pixels_200smoke1800nosmoke']
for fileNumber,fn in enumerate(fns):
    print('开始训练'+fn.split('\\')[-1])
    num_layer = eval(fn.split('\\')[-1].split('_')[1])
    hid_vertex = eval(fn.split('\\')[-1].split('_')[0])
    td_pth = glob.glob(r'{}\*\*_td.npy'.format(rootDataset[fileNumber]))#I:\MLP_RESULT\trainging_pixel_npy_0129
    vd_pth =glob.glob(r'{}\*\*_vd.npy'.format(rootDataset[fileNumber]))
    train(td_pth,vd_pth,fn,[1,2,3,13,14,15,16],2,hid_vertex,num_layer,500,1000,1e-3)
    #train函数的参数从左到右分别为：训练数据的路径及文件名；验证数据的路径及文件名；训练结果保存路径；训练数据选择的通道号码；
    #网络输出层节点数；网络隐层各层节点数；网络隐层层数；训练过程的最大迭代次数；一次反传所包含的批量训练样本数；网络学习率
    #train(td_path, vd_path, result_path, input_channels, output_vertex, hidden_vertex, num_layer, echo, batchsize, lr)
print('跑完了')
