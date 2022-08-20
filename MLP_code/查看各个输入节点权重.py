import os
import shutil
import time
import numpy as np
import torch
import gdal
import glob
from FCN import FCN

def query_weights(net_file, net_para, bands):
    '''
        net_file = 文件存储路径包括名字
        net_para = 网络参数
        bands = 选择的葵花影像和风数据通道，葵花数据是1-16通道，风数据是17，18通道，标签是19通道
    '''
    def load_network():
        # 加载网络，输出的是训练好的网络模型
        (input_vertex, output_vertex, hidden_vertex, num_layer, BN, Drop) = net_para
        net = FCN(input_vertex, output_vertex, hidden_vertex, num_layer,BN, Drop).cuda()  #
        net.load_state_dict(torch.load(net_file))  # map_location=torch.device('cpu')
        net = net.eval()
        return net
    def calculate_input(net):
        lst = []
        copy = torch.Tensor()
        for para in net.parameters():
            for i in range(len(bands)):
                copy = para
                copy=copy.abs()
                lst.append(copy[:,i].cpu().detach().numpy().mean())
            break #没辙只能用for循环看
        print(lst)
        return lst
    def display_order(lst):
        #显示一个一维数组的序号
        bigtosmall = sorted(lst)
        bigtosmall.reverse()
        output = []
        for value in lst:
            output.append(bigtosmall.index(value)+1)
        print(output)
    net = load_network()
    l=calculate_input(net)
    display_order(l)
net_file = r'E:\SmokeDetection\sink\MLP\Noon256batchsize3e4lrBN050Dropout\fcn_297_loss.pth'
#r'E:\SmokeDetection\source\MLP_NW_exp\4000_smoke\6\fcn_55_f1.pth'
#r'E:\SmokeDetection\sink\MLP\Noon256batchsize3e4lrBN050Dropout\fcn_297_loss.pth'
bands = [1,2,3,7,11,13,14,15,16]
net_para = (len(bands), 2, 256, 3, 0.5, 0)
query_weights(net_file, net_para,bands)
print(bands)

'''
根据实验结果，4000smoke中可见光波段的权值序列很低，50000的序列很高
可见光的地位被增强了
4000：
[0.18302128, 0.18562898, 0.22706376, 0.18864939, 0.20733914, 0.21417266, 0.19345075, 0.21372238, 0.20762247]
波段地位：[9, 8, 1, 7, 5, 2, 6, 3, 4]
波段序列：[1, 2, 3, 7, 11, 13, 14, 15, 16]

波段地位表示这个波段的重要性，比如第一波段是9，说明它在第9位；第七波段是第七位，说明它影响力第七

50000：
[0.64910316, 0.44617617, 0.64852726, 0.35798457, 0.42821944, 0.59087783, 0.42088404, 0.5004266, 0.62203664]
波段地位：[1, 6, 2, 9, 7, 4, 8, 5, 3]
波段序列：[1, 2, 3, 7, 11, 13, 14, 15, 16]

通过比较权重的平均值的排序，我知道了增加样本可以提高可见光波段的地位
'''