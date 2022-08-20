from batch_train import train
import glob
import random
import os
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


#不同数据集相同框架的MLP的存储路径，命名形式:每幅图采样像元数_采样方式\随机次数
fns = glob.glob(r'E:\SmokeDetection\source\MLP_cirrus_exp\50000_smokemorning\Morning_20220811_2')#结果存储

#开始训练，根据文件夹上标注的层数和节点数训练
rootDataset = glob.glob(r'E:\SmokeDetection\source\MLP_cirrus_test\50000_smoke2500\Morning_20220811_2')#数据所在地
for fileNumber,fn in enumerate(fns):
    print('开始训练'+fn.split('\\')[-2]+'_'+fn.split('\\')[-1])
    num_layer = 3 #eval(fn.split('\\')[-1].split('_')[1])
    hid_vertex = 256 #eval(fn.split('\\')[-1].split('_')[0])
    td_pth = glob.glob(r'{}\*\*_td.npy'.format(rootDataset[fileNumber]))#I:\MLP_RESULT\trainging_pixel_npy_0129
    vd_pth =glob.glob(r'{}\*\*_vd.npy'.format(rootDataset[fileNumber]))
    #print(vd_pth)
    filesize = eval(rootDataset[fileNumber].split('\\')[-2].split('_')[-2])
    train(td_pth,vd_pth,fn,[1,2,6,7,11,13],2,hid_vertex,num_layer,100,20000,3e-4,filesize,BN=True, Drop=0.25)
    #train函数的参数从左到右分别为：训练数据的路径及文件名；验证数据的路径及文件名；训练结果保存路径；训练数据选择的通道号码；
    #网络输出层节点数；网络隐层各层节点数；网络隐层层数；训练过程的最大迭代次数；一次反传所包含的批量训练样本数；网络学习率；是否在激活函数后加BN层；是否在BN层后加Dropout，不加默认0
    #train(td_path, vd_path, result_path, input_channels, output_vertex, hidden_vertex, num_layer, echo, batchsize, lr)[1,2,3,7,11,13,14,15,16]

print('跑完了')
