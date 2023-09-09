#分配样本进原始训练数据集中，初始化的样本不在此分配
import os
import shutil
import time
import numpy as np
import torch
import gdal
from FCN import FCN

def data_assgin(net_file, net_para, samples_files, td_pth, vd_pth, num):
    '''
    将总数据集中错误率高的未在训练集中的样本分入训练集
    net_file是神经网络文件路径名
    net_para是网络的参数，元组，包含如下才参数
    (input_vertex, output_vertex, hidden_vertex, num_layer)
    输入节点数、输出节点数、每个隐藏层节点数、隐藏层数
    samples_files是总样本路径名
    td_pth,vd_pth是样本集（包含训练和验证集）的文件名索引的txt文件
    num是想分配进训练集中的样本数量
    仅支持MLP
    '''
    os.chdir(samples_files)
    def load_network():
        (input_vertex, output_vertex, hidden_vertex, num_layer) = net_para
        net = FCN(input_vertex, output_vertex, hidden_vertex, num_layer).cuda() #
        net.load_state_dict(torch.load(net_file))
        return net

    def load_smp_lst(td_pth, vd_pth, jdg=1):
        '''
        加载已有的训练数据和验证数据索引列表
        td_pth/vd_pth是训练和验证集的txt名字索引
        jdg为0的话就不会启动这个程序，算是一个保险吧。
        '''
        if jdg:
            with open(td_pth, 'r') as f:
                tra_data = f.readlines()
                print('成功加载已有训练集')
            with open(vd_pth, 'r') as f:
                val_data = f.readlines()
                print('成功加载已有验证集')
            vd = []
            td = []
            for t in tra_data:
                td.append(t.strip('\n'))
            for v in val_data:
                vd.append(v.strip('\n'))
            return td, vd

    def load_data(td,vd):
        #遍历所有没有在训练数据集里的数据，读取它们的文件路径，索引成序列
        idx = os.listdir()
        fns = td+vd #这里我将验证集也加了进去，因为我想趁此机会筛选掉一些烟，然后专注于对云的识别
        for fn in fns:
            idx.remove(fn)
        return idx


    def prediction(idx, net):
        batch_size = 1
        # 所有load_data序列中的文件经过网络，输出OA序列
        def cal_oa(pred_choice, target):
            # 计算batch_size*4096个样本的平均精确度
            TP, TN, FP, FN = 0, 0, 0, 0
            # TP    predict 和 label 同时为1
            TP += ((pred_choice == 1) & (target.data == 1)).sum().float()
            # TN    predict 和 label 同时为0
            TN += ((pred_choice == 0) & (target.data == 0)).sum().float()
            # FN    predict 0 label 1
            FN += ((pred_choice == 0) & (target.data == 1)).sum().float()
            # FP    predict 1 label 0
            FP += ((pred_choice == 1) & (target.data == 0)).sum().float()

            acc = (TP + TN) / (TP + TN + FP + FN)
            return acc

        OA = []
        c =0
        for file_num in range(0, len(idx), batch_size):
            for count, file in enumerate(idx[file_num:file_num + batch_size]):
                ary = gdal.Open(file).ReadAsArray().reshape([19, 64 * 64]).T  # ary.shape=[4096,19]
                if count == 0:
                    data = ary[:, :16]  # [4096,18] 不包含风
                    label = ary[:, 18]  # [4096,1]
                else:
                    #data = np.r_[data, ary[:, :18]] #包含风
                    data = np.r_[data, ary[:, :16]] #不包含风
                    label = np.r_[label, ary[:, 18]]
                # 进行预处理操作之后，输出一个64*64的小图的OA
                data = torch.from_numpy(data).float().cuda()  # size=[4096*batch size,18]
                label = torch.from_numpy(label).long().cuda()
                pred = net(data)
                prediction = torch.max(pred, 1)[1]
                print(prediction.cpu())
                oa = cal_oa(prediction.cpu(), label.cpu())
                OA.append(oa)
                c += 1
                print('成功输出第{}张图的OA--{}'.format(c,oa))
        return OA

    def sieve_samples(idx, OA):
        # 筛选出num个数量的错误率最大的样本，以序列形式输出
        #2020.11.12 这代码写的真烂
        dictionary = []
        for i in range(len(idx)):
            dictionary.append({'name':idx[i],'accuracy':OA[i]})#字典序列
        d = sorted(dictionary, key=lambda dd: dd['accuracy'])
        out_idx = []
        OA = []
        for j in range(num):
            out_idx.append(d[j]['name'])
        for i in range(num):
            OA.append(d[i]['accuracy'])
        return out_idx,OA

    def output_files(out_idx, accuracy):
        '''
        输出分配数据的索引
        输出这些数据对应的OA
        '''

        with open(r'E:\SmokeDetection\source\MLP\document'+os.sep+r'add_smp_list.txt','a') as f:  #懒得做接口了，给你索引输进这个文件夹里
            f.writelines(p+'\n' for p in out_idx)
            print('成功输出索引数据{}个'.format(len(out_idx)))
        with open(r'E:\SmokeDetection\source\MLP\document'+os.sep+r'add_OA_list.txt','a') as f:  #懒得做接口了，给你OA输进这个文件夹里
            f.writelines(str(p.numpy())+'\n' for p in accuracy)
            print('成功输出精度数据{}个'.format(len(accuracy)))

    net = load_network()
    print('成功加载网络')
    td,vd = load_smp_lst(td_pth, vd_pth)
    idx = load_data(td,vd)
    '''with open(r'E:\SmokeDetection\source\MLP\document\add_smp_list1.txt', 'r') as f:
        tra_data = f.readlines()
        print('成功加载已有训练集')
        idx= []
        for t in tra_data:
            idx.append(t.strip('\n'))'''
    print('成功加载非训练数据')
    OA = prediction(idx, net)
    print('成功加载精度索引')
    out_idx,acc = sieve_samples(idx,OA)
    print('成功加载输出数据索引')
    output_files(out_idx,acc)


if __name__ == '__main__':
    net_file = r'E:\SmokeDetection\source\MLP\result\72_10\fcn_1888.pth' #网络的文件
    input_vertex, output_vertex, hidden_vertex, num_layer = 16, 2, 72, 10 #网络的参数
    net_para = [input_vertex, output_vertex, hidden_vertex, num_layer]
    samples_files = r'E:\SmokeDetection\source\new_data\samplesfortrain_2020_10_25'  #总样本路径，训练数据取自其中
    td_pth = r'E:\SmokeDetection\source\MLP\document\training_smp_list_NW.txt'  #训练数据的文件名索引文件
    vd_pth = r'E:\SmokeDetection\source\MLP\document\validation_smp_list_NW.txt' #验证数据的文件名索引文件
    num = 200  #想新增多少个样本进去训练集中
    data_assgin(net_file, net_para, samples_files, td_pth,vd_pth, num)
    #最终会输出2个文件，一个包含要加入的文件名索引，另一个包含这些文件的OA