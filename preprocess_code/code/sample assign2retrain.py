#分配样本进原始训练数据集中，初始化的样本不在此分配
import os
import shutil
import time
import glob
import numpy as np
import torch
import gdal
from FCN import FCN

def data_assgin(net_file, net_para, samples_files, training_samples_file, num, net_name):
    '''
    将总数据集中错误率高的未在训练集中的样本分入训练集
    net_file是神经网络文件路径名
    net_para是网络的参数，元组，包含如下才参数
    (input_vertex, output_vertex, hidden_vertex, num_layer)
    输入节点数、输出节点数、每个隐藏层节点数、隐藏层数
    samples_files是总样本路径名
    training_samples_file是训练样本集（包含训练和验证集）的文件夹路径
    num是想分配进训练集中的样本数量
    net_name时网络的类型，有'MLP','UNET'2种。
    '''
    def load_network():
        (input_vertex, output_vertex, hidden_vertex, num_layer) = net_para
        net = FCN(input_vertex, output_vertex, hidden_vertex, num_layer) #
        net.load_state_dict(torch.load(net_file))
        return net

    def load_data(net):
        #遍历所有没有在训练数据集里的数据，读取它们的文件路径，索引成序列
        idx = []
        fns = glob.glob(training_samples_file+r'\*.tif')
        for fn in samples_files:
            if not fn in fns:
                idx.append(fn)
        return idx

    def prediction(idx, net):
        batch_size = 1
        # 所有load_data序列中的文件经过网络，输出OA序列
        def cal_oa(pred_choice, target):
            # 计算batch_size*4096个样本的平均精确度
            TP, TN, FP, FN = 0, 0, 0, 0
            # TP    predict 和 label 同时为1
            TP += ((pred_choice == 1) & (target.data == 1)).cpu().sum()
            # TN    predict 和 label 同时为0
            TN += ((pred_choice == 0) & (target.data == 0)).cpu().sum()
            # FN    predict 0 label 1
            FN += ((pred_choice == 0) & (target.data == 1)).cpu().sum()
            # FP    predict 1 label 0
            FP += ((pred_choice == 1) & (target.data == 0)).cpu().sum()
            return (TP + TN) / (TP + TN + FP + FN)

        OA = []
        for file_num in range(0, len(idx), batch_size):
            for count, file in enumerate(idx[file_num:file_num + batch_size]):
                ary = gdal.Open(file).ReadAsArray().reshape([19, 64 * 64]).T  # ary.shape=[4096,19]
                if count == 0:
                    data = ary[:, :18]  # [4096,18]
                    label = ary[:, 18]  # [4096,1]
                else:
                    data = np.r_[data, ary[:, :18]]
                    label = np.r_[label, ary[:, 18]]
                # 进行预处理操作之后，输出一个64*64的小图的OA
                data = torch.from_numpy(data).float().cuda()  # size=[4096*batch size,18]
                label = torch.from_numpy(label).long().cuda()
                pred = net(data)
                prediction = torch.max(pred, 1)[1]
                oa = cal_oa(prediction, label)
                OA.append(oa)
        return OA

    def sieve_samples(OA, idx):
        # 筛选出num个数量的错误率最大的样本，以序列形式输出
        dictionary = []
        for i in range(len(idx)):
            dictionary.append({'name':idx[i],'accuracy':OA[i]})#字典序列
        d = sorted(dictionary, key=lambda dd: dd['accuracy'], reverse=True)
        out_idx = []
        for j in range(num):
            out_idx.append(d[j]['name'])
        return out_idx

    def assign_samples(out_idx):
        '''分配数据，顺便输出分配数据的索引'''
        for fn in out_idx:
            shutil.copy(fn, training_samples_file)
        with open(training_samples_file+r'add_smp_list.txt','a') as f:
            f.writelines(p+'\n' for p in out_idx)



    net = load_network()
    idx = load_data(net)
    OA = prediction(idx, net)
    out_idx = sieve_samples(OA, idx)
    assign_samples(out_idx)



if __name__ == '__main__':

    def load_network(net_file, net_para):
        (input_vertex, output_vertex, hidden_vertex, num_layer) = net_para
        net = FCN(input_vertex, output_vertex, hidden_vertex, num_layer) #
        net.load_state_dict(torch.load(net_file))
        return net
    def prediction(idx, net):
        batch_size =1
        # 所有load_data序列中的文件经过网络，输出OA序列
        def cal_oa(pred_choice, target):
            #计算batch_size*4096个样本的平均精确度
            TP ,TN, FP, FN = 0,0,0,0
            # TP    predict 和 label 同时为1
            TP += ((pred_choice == 1) & (target.data == 1)).cpu().sum()
            # TN    predict 和 label 同时为0
            TN += ((pred_choice == 0) & (target.data == 0)).cpu().sum()
            # FN    predict 0 label 1
            FN += ((pred_choice == 0) & (target.data == 1)).cpu().sum()
            # FP    predict 1 label 0
            FP += ((pred_choice == 1) & (target.data == 0)).cpu().sum()
            return (TP+TN)/(TP+TN+FP+FN)

        OA=[]
        for file_num in range(0, len(idx), batch_size):
            for count, file in enumerate(idx[file_num:file_num + batch_size]):
                ary = gdal.Open(file).ReadAsArray().reshape([19, 64 * 64]).T  # ary.shape=[4096,19]
                if count == 0:
                    data = ary[:, :18]  # [4096,18]
                    label = ary[:, 18]  # [4096,1]
                else:
                    data = np.r_[data, ary[:, :18]]
                    label = np.r_[label, ary[:, 18]]
                # 进行预处理操作之后，输出一个64*64的小图的OA
                data = torch.from_numpy(data).float().cuda()  # size=[4096*batch size,18]
                label = torch.from_numpy(label).long().cuda()
                pred = net(data)
                prediction = torch.max(pred, 1)[1]
                oa = cal_oa(prediction,label)
                OA.append(oa)
        return OA
    net = FCN(18).cuda()
    '''
    torch.save(net.state_dict(), r'I:\烟检测论文写作\tests_workplace\test.pth')
    n=load_network(r'I:\烟检测论文写作\tests_workplace\test.pth', (16,2,32,2))
    d = [p for p in net.named_parameters()]
    print(d)
    '''#测试load_net的代码
    fns = glob.glob(r'I:\samples_2020_10_25\*.tif')
    idx = fns[:2]
    print(time.asctime())
    oa = prediction(idx, net)
    print(time.asctime())

