#大张旗鼓，半监督；做不出来，就是猪
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


#不同数据集相同框架的MLP的存储路径，命名形式:每幅图采样像元数_采样方式\随机次数
'''smp_rlt = '0040+0140-真标签'
fn = r'E:\SmokeDetection\source\semi-supervised learning\Results\{}'.format(smp_rlt)#结果存储
pth = '0050+0150-' #基于的模型 已经失效0804 10：06
td_pth = glob.glob(r'E:\SmokeDetection\source\semi-supervised learning\Samples0825\{}\*_td.npy'.format(smp_rlt))#制作采样的样本
vd_pth =glob.glob(r'E:\SmokeDetection\source\semi-supervised learning\Samples0825\{}\*_vd.npy'.format(smp_rlt))
net_file = glob.glob(r'E:\SmokeDetection\source\semi-supervised learning\Results\{}\*loss.pth'.format(pth))[0]
#开始训练，根据文件夹上标注的层数和节点数训练
'''
#20210913
fn = r'E:\SmokeDetection\source\semi-supervised learning\result\test 0830 2hours with time semi065 small'#结果存储
smp_pth = r'E:\SmokeDetection\source\semi-supervised learning\manual_samples\0830' #样本存储
td_pth = glob.glob(r'{}\*_td.npy'.format(smp_pth))[6:18]#制作采样的样本
vd_pth = glob.glob(r'{}\*_vd.npy'.format(smp_pth))[6:18]#glob.glob(r'E:\SmokeDetection\source\semi-supervised learning\manual_samples\0825\*_vd.npy')[:12]

print('开始训练')
num_layer = 1 #eval(fn.split('\\')[-1].split('_')[1])
hid_vertex = 17 #eval(fn.split('\\')[-1].split('_')[0])
semi_pth = glob.glob(r'E:\SmokeDetection\source\semi-supervised learning\cliped_new_new_data\0830\00*0.tif')[:]# tif文件string格式组成的List
#print(vd_pth)
filesize = 220*230
train(td_pth,vd_pth,fn,[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17],2,hid_vertex,num_layer,100,128,5e-3,filesize,True, 0,1,semi_pth)
#train函数的参数从左到右分别为：训练数据的路径及文件名；验证数据的路径及文件名；训练结果保存路径；训练数据选择的通道号码；
#网络输出层节点数；网络隐层各层节点数；网络隐层层数；训练过程的最大迭代次数；一次反传所包含的批量训练样本数；网络学习率；
# 一张图里采样的训练样本数；是否在激活函数后加BN层；是否在BN层后加Dropout，不加默认0；现有网络的名称
#train(td_path, vd_path, result_path, input_channels, output_vertex, hidden_vertex, num_layer, echo, batchsize, lr)[1,2,3,7,11,13,14,15,16]
#td_path, vd_path, result_path, bands, output_vertex, hidden_vertex, num_layer, echo, batchsize, lr, filesize,BN,Drop,net_file
print('跑完了')
