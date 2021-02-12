from FCN import FCN
import torch
import os
import numpy as np
import glob
import gdal
import time
import random
from focalloss import FocalLoss as focal


def find_pix_index(pix, fn):
    #pix包含了一个像元的所有数据，19个波段
    date = fn.split('\\')[-2]
    t = fn.split('\\')[-1][:4]
    fn = r'E:\SmokeDetection\source\new_new_data\{}\{}.tif'.format(date,t)
    ary = gdal.Open(fn).ReadAsArray()
    for i in range(501):
        for j in range(582):
            idx = ary[:,i,j] == pix
            if idx.sum()==19:
                print(pix)
                with open(r'E:\SmokeDetection\source\t_place\wrong_idx.txt','a') as f:
                    f.writelines(str(fn)+str([i, j])+'  '+str(pix[-1])+'\n')
                    f.writelines(str(pix)+'\n')

def load_network(net_para, net_file):
    # 加载网络，输出的是训练好的网络模型
    # net_para是网络参数
    (input_vertex, output_vertex, hidden_vertex, num_layer) = net_para
    net = FCN(input_vertex, output_vertex, hidden_vertex, num_layer).cuda()  #自己写的FCN代码
    net.load_state_dict(torch.load(net_file))
    return net

def cal_evaluation(TP, TN, FP, FN):
    # 只需要输入混淆矩阵四个值，输出各种评价参数
    T = TP + TN + FP + FN
    acc = (TP + TN) / T
    pre = TP / (TP + FP + 1e-8)
    rcl = TP / (TP + FN + 1e-8)
    iou = TP / (TP + FN + FP + 1e-8)
    fone = 2 * pre * rcl / (pre + rcl + 1e-8)
    temp = ((TP + FP) * (TP + FN) + (TN + FP) * (TN + FN)) / (T * T)
    kpa = (acc - temp) / (1 - temp+1e-9)
    return acc, pre, rcl, iou, fone, kpa
def normalize(a):
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
    ary = np.copy(a)
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

def train(td_path, vd_path, result_path, bands, output_vertex, hidden_vertex, num_layer, echo, filesize, lr):
    #bands指的是输入的通道序号，葵花卫星短到长波为从1到16，风数据从17到18，标签数据19（-1）
    print(time.asctime())
    input_vertex = len(bands)
    os.chdir(result_path)
    mn = None
    temp = 0
    net = FCN(input_vertex, output_vertex, hidden_vertex, num_layer).cuda()
    #net = load_network((input_vertex, output_vertex, hidden_vertex, num_layer), r'E:\SmokeDetection\source\MLP_Results\fcn_261.pth')
    #opt = torch.optim.Adam(params=net.parameters(),lr=lr) #BatchSize=2000+
    opt = torch.optim.SGD(params=net.parameters(),momentum=0.9,lr=lr)#weight_decay=L2正则
    for e in range(echo):
        iou_td = 0
        iou_vd = 0  # iou=tp/(tp+fp+fn)
        accuracy_td = 0
        accuracy_vd = 0
        loss_td = 0
        loss_vd = 0
        recall_td = 0  # recall查全 precision 查准
        recall_vd = 0
        precision_td = 0
        precision_vd = 0

        val_data_num = 0
        tp, tn, fp, fn = 0, 0, 0, 0
        TP, TN, FP, FN = 0, 0, 0, 0
        fns = td_path #接口改在外面了
        random.shuffle(fns)
        #fns = [r'E:\SmokeDetection\source\new_samples_64\0823\0000_1344_1088_smoke.tif'] ##!!!
        td_back_time = 0#每个周期的反向传播次数
        for file_num in range(0,len(fns),filesize):
            for count,file in enumerate(fns[file_num:file_num+filesize]):
                ary1 = np.load(file)
                ary = normalize(ary1)
                np.random.shuffle(ary)
                #ary = np.random.random(ary.shape) #造假数据看看
                '''
                ary_temp = np.zeros([19])
                for ii in range(64*64):
                    if ary[ii,-1] == 1:
                        ary_temp=np.c_[ary[ii],ary_temp]
                ary = ary_temp.T
                '''
                if count==0:
                    data = ary[:,:18]  #[4096,18] 考虑风
                    #data = ary[:, :16]
                    label = ary[:,18]  #[4096,1]
                else:
                    data = np.r_[data,ary[:,:18]] #########考虑风
                    #data = np.r_[data, ary[:, :16]]  ###########不考虑风
                    label = np.r_[label,ary[:,18]]
            data = choose_bands(data, bands) #选择加入训练的波段
            #print(np.where(label==1))
            input = torch.from_numpy(data).float().cuda()  #size=[750*batch size,16]
            gt = torch.from_numpy(label).long().cuda()
            opt.zero_grad()
            pred = net(input)
            #L1 正则
            if 0:
                regL1 = 0
                for para in net.parameters():
                    regL1 += torch.sum(torch.abs(para))
                loss = torch.nn.CrossEntropyLoss()(pred, gt) + 0.01*regL1  # pred.size=[750,2]
            loss = torch.nn.CrossEntropyLoss()(pred, gt)
            td_back_time += 1
            loss_td += loss
            loss.backward()
            opt.step()
            prediction = torch.max(pred,1)[1].cpu().numpy()
            gt = gt.cpu().numpy()

            for i in range(gt.shape[0]):
                if gt[i]==1:
                    if prediction[i]==1:
                        tp+=1
                    if prediction[i]==0:
                        fn+=1
                        #find_pix_index(ary1[i,:], file)
                elif gt[i]==0:
                    if prediction[i]==0:
                        tn+=1
                    if prediction[i]==1:
                        fp+=1
                        #find_pix_index(ary1[i,:], file)
                else :print('这有问题：'+gt[i]+ary1[i])
            del ary
        del input, gt, pred, loss
        accuracy_td,precision_td,recall_td,iou_td,fone_td,kpa_td=cal_evaluation(tp,tn,fp,fn)
        loss_td /= td_back_time
        print('echo={},finish train time={}'.format(e,time.asctime()))
        os.chdir(result_path)
        with open('accuracy_td.txt', 'a') as f:
            f.write('{}\n'.format(accuracy_td))
        with open('loss_td.txt', 'a') as f:
            f.write('{}\n'.format(loss_td))
        with open('recall_td.txt', 'a') as f:
            f.write('{}\n'.format(recall_td))
        with open('precision_td.txt', 'a') as f:
            f.write('{}\n'.format(precision_td))
        with open('kappa_td.txt', 'a') as f:
            f.write('{}\n'.format(kpa_td))
        with open('fone_td.txt', 'a') as f:
            f.write('{}\n'.format(fone_td))
        with open('iou_td.txt', 'a') as f:
            f.write('{}\n'.format(iou_td))


        print(accuracy_td)
        ##################开始验证################validation####
        val_fns = vd_path
        vd_back_time =0
        for file_num in range(0, len(val_fns), filesize):

            for count, file in enumerate(val_fns[file_num:file_num + filesize]):
                ary = np.load(file) # shape=[250,19]
                ary = normalize(ary)
                if count == 0:
                    data = ary[:, :18]  # [750,18]考虑风
                    #data = ary[:, :16]  #不考虑风
                    label = ary[:, 18]  # [750,1]
                else:
                    data = np.r_[data, ary[:, :18]] #考虑风
                    #data = np.r_[data, ary[:, :16]] #不考虑风
                    label = np.r_[label, ary[:, 18]]
            data = choose_bands(data,bands)
            input_vd = torch.from_numpy(data).float().cuda()  # size=[4096*batch size,18]
            gt = torch.from_numpy(label).long().cuda()
            pred_vd = net(input_vd)
            #torch.cuda.empty_cache()
            loss = torch.nn.CrossEntropyLoss()(pred_vd, gt)  # pred.size=[4096,2]
            #loss = loss.detach()
            vd_back_time += 1
            loss_vd += loss
            prediction = torch.max(pred_vd,1)[1].cpu().numpy()
            #input_vd.detach()
            gt = gt.cpu().numpy()
            for i in range(gt.shape[0]):
                if gt[i] == 1:
                    if prediction[i] == 1:
                        TP += 1
                    if prediction[i] == 0:
                        FN += 1
                if gt[i] == 0:
                    if prediction[i] == 0:
                        TN += 1
                    if prediction[i] == 1:
                        FP += 1
            del input_vd,gt,pred_vd,loss
        print('echo={},finish validation time={}'.format(e, time.asctime()))
        accuracy_vd,precision_vd,recall_vd,iou_vd,fone_vd,kpa_vd=cal_evaluation(TP,TN,FP,FN)
        loss_vd /= vd_back_time
        with open('accuracy_vd.txt','a') as f:
            f.write('{}\n'.format(accuracy_vd))
        with open('loss_vd.txt','a') as f:
            f.write('{}\n'.format(loss_vd))
        with open('recall_vd.txt','a') as f:
            f.write('{}\n'.format(recall_vd))
        with open('precision_vd.txt','a') as f:
            f.write('{}\n'.format(precision_vd))
        with open('fone_vd.txt','a') as f:
            f.write('{}\n'.format(fone_vd))
        with open('iou_vd.txt', 'a') as f:
            f.write('{}\n'.format(iou_vd))
        with open('kappa_vd.txt', 'a') as f:
            f.write('{}\n'.format(kpa_vd))

        if mn is None:
            mn = 'fcn_{}.pth'.format(e)
            torch.save(net.state_dict(), mn)
        if temp < iou_vd: ##############################注意啊这里被改成看验证数据的iou了
            os.remove(mn)
            mn = 'fcn_{}.pth'.format(e)
            torch.save(net.state_dict(), mn)
            temp = iou_vd
        print(time.asctime())



