#这是将已有的模型载入，模型识别有问题的像元值会对应到图像上的代码
import gdal
import numpy as np
import glob
import torch
import torch.nn as nn
from FCN import FCN

def normalize(ary):
    '''
    这是给7-16波段葵花数据以及17-18波段风数据进行一个拉伸，葵花亮温数据拉伸到0-1，经纬向风数据拉伸到[-1,1]
    1-18通道的五天所有数据的最小值：
    [ 4.49999981e-03  3.59999994e-03  2.19999999e-03  3.99999990e-04
      0.00000000e+00  0.00000000e+00  2.02559998e+02  1.88289993e+02
      1.84699997e+02  1.86860001e+02  1.85389999e+02  2.09250000e+02
      1.85529999e+02  1.83789993e+02  1.84559998e+02  1.86830002e+02
     -9.23541546e+00 -2.66013598e+00]
     1-18通道的五天所有数据的最大值：
    [  1.21569991   1.21569991   1.21569991   1.2184       1.21569991
       1.21569991 400.13000488 250.29998779 260.88000488 269.8500061
     313.47998047 287.67999268 316.47998047 313.77999878 305.1000061
     283.45999146   7.96795177   8.07158279]
    '''
    for i in range(ary.shape[1]):
        if 5 >= i >= 0:
            ary[:,i] = (ary[:,i]-0)/(1.22-0)
        if 15 >= i >= 6:
            ary[:,i] = (ary[:,i]-180)/(401-180)
        if 17>= i >= 16:
            ary[:, i] = ary[:, i] / 10
    return ary

def choose_bands(data,bands):
    '''
    这是选择对应波段加入神经网络，
    bands是一个列表，里面的数字代表选取对应波段，使用的是人类语言，比如选择1，2，3波段输入网络就是bands=[1,2,3]
    '''
    output = data[:,bands[0]-1] #shape=750
    for band in bands[1:]:
        output = np.c_[output,data[:,band-1]]
    return output

def load_network(net_para, net_file):
    # 加载网络，输出的是训练好的网络模型
    # net_para是网络参数
    (input_vertex, output_vertex, hidden_vertex, num_layer) = net_para
    net = FCN(input_vertex, output_vertex, hidden_vertex, num_layer).cuda()  #自己写的FCN代码
    net.load_state_dict(torch.load(net_file))
    return net

def find_pix_index(pix, fn):
    #pix包含了一个像元的所有数据，19个波段
    ary = gdal.Open(fn).ReadAsArray()
    for i in range(501):
        for j in range(582):
            idx = ary[:,i,j] == pix
            if idx.sum()==19:
                with open(r'E:\SmokeDetection\source\t_place\wrong_idx.txt','a') as f:
                    f.writelines(str(fn)+str([i, j])+'\n')

if __name__ == '__main__':

    net_file = r'E:\SmokeDetection\source\MLP_Results\fcn_261.pth'
    net_para = (10, 2, 1024, 10)
    net = load_network(net_para, net_file)

    find_pix_index(pix,fn)
