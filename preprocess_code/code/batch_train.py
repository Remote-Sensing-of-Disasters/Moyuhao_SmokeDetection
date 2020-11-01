from FCN import FCN
import torch
import os
import numpy as np
import glob
import gdal
import time
import random
from focalloss import FocalLoss as focal


def train(td_path, vd_path, result_path, input_vertex, output_vertex, hidden_vertex, num_layer, echo, batchsize, lr):
    print(time.asctime())
    os.chdir(result_path)
    mn = None
    temp = 100000
    net = FCN(input_vertex, output_vertex, hidden_vertex, num_layer).cuda()
    opt = torch.optim.Adam(params=net.parameters(),lr=lr)
    #opt = torch.optim.SGD(params=net.parameters(),momentum=0.9,lr=lr)
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
        fns = glob.glob(td_path+r'\0817\*.tif')##############
        #fns = [r'E:\SmokeDetection\source\new_samples_64\0823\0000_1344_1088_smoke.tif'] ##!!!
        for file_num in range(0,len(fns),batchsize):
            for count,file in enumerate(fns[file_num:file_num+batchsize]):
                ary = gdal.Open(file).ReadAsArray().reshape([19,64*64]).T  #ary.shape=[4096,19]

                '''
                ary_temp = np.zeros([19])
                for ii in range(64*64):
                    if ary[ii,-1] == 1:
                        ary_temp=np.c_[ary[ii],ary_temp]
                ary = ary_temp.T
                '''
                if count==0:
                    data = ary[:,:18]  #[4096,18]
                    label = ary[:,18]  #[4096,1]
                else:
                    data = np.r_[data,ary[:,:18]]
                    label = np.r_[label,ary[:,18]]
            #print(np.where(label==1))

            input = torch.from_numpy(data).float().cuda()  #size=[4096*batch size,18]
            gt = torch.from_numpy(label).long().cuda()


            #input = input[2586:2606]#2599-2623是烟
            #gt = gt[2584:2604]
            #print(gt.sum())
            #print(gt.size())


            opt.zero_grad()
            pred = net(input)
            loss = torch.nn.CrossEntropyLoss()(pred, gt)  # pred.size=[4096,2]
            '''
            para=net.parameters
            d= [p.numel() for p in net.parameters()]
            print(d)
            print(net.parameters)
            
            '''
            #loss = focal()(pred,gt)
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
                if gt[i]==0:
                    if prediction[i]==0:
                        tn+=1
                    if prediction[i]==1:
                        fp+=1
            del ary
        del input, gt, pred, loss
        iou_td = tp / (tp + fp + fn + 1e-9)
        accuracy_td = (tp + tn) / (tp + fp + fn + tn)
        precision_td = tp / (tp + fp + 1e-9)
        recall_td = tp / (tp + fn + 1e-9)
        loss_td /= len(fns)
        print('echo={},finish train time={}'.format(e,time.asctime()))
        print(tp)
        print(tn)
        print(fp)
        print(fn)
        #print(loss)
        print(e)
        os.chdir(result_path)
        with open('accuracy_td.txt', 'a') as f:
            f.write('{}\n'.format(accuracy_td))
        with open('loss_td.txt', 'a') as f:
            f.write('{}\n'.format(loss_td))
        with open('recall_td.txt', 'a') as f:
            f.write('{}\n'.format(recall_td))
        with open('precision_td.txt', 'a') as f:
            f.write('{}\n'.format(precision_td))

        val_fns = glob.glob(vd_path+r'\*\*.tif')

        for file_num in range(0, len(val_fns), batchsize):
            for count, file in enumerate(val_fns[file_num:file_num + batchsize]):
                ary = gdal.Open(file).ReadAsArray().reshape([19, 64 * 64]).T  # shape=[4096,19]
                if count == 0:
                    data = ary[:, :18]  # [4096,18]
                    label = ary[:, 18]  # [4096,1]
                else:
                    data = np.r_[data, ary[:, :18]]
                    label = np.r_[label, ary[:, 18]]
            val_data_num = count + 1
            input = torch.from_numpy(data).float().cuda()  # size=[4096*batch size,18]
            gt = torch.from_numpy(label).long().cuda()
            pred = net(input)
            loss = torch.nn.CrossEntropyLoss()(pred, gt)  # pred.size=[4096,2]
            loss_vd += loss
            prediction = torch.where(pred[:, 0] < pred[:, 1], torch.ones(gt.size()[0]).cuda(),
                                     torch.zeros(gt.size()[0]).cuda()).cpu().numpy()

            gt = gt.cpu().numpy()
            for i in range(gt.shape[0]):
                if prediction[i] > label[i]:
                    FP += 1
                if prediction[i] < label[i]:
                    FN += 1
                if prediction[i] == label[i] == 0:
                    TN += 1
                if prediction[i] == label[i] == 1:
                    TP += 1
        del input,gt,pred,loss
        print('echo={},finish validation time={}'.format(e, time.asctime()))
        iou_vd = TP / (TP + FP + FN +1e-9)
        accuracy_vd = (TP + TN) / (TP + FP + FN + TN)
        precision_vd = TP/ (TP + FP + 1e-9)
        recall_vd = TP/ (TP + FN + 1e-9)
        #loss_vd /= val_data_num
        
        
        with open('accuracy_vd.txt','a') as f:
            f.write('{}\n'.format(accuracy_vd))
        
        with open('loss_vd.txt','a') as f:
            f.write('{}\n'.format(loss_vd))
        
        with open('recall_vd.txt','a') as f:
            f.write('{}\n'.format(recall_vd))
        
        with open('precision_vd.txt','a') as f:
            f.write('{}\n'.format(precision_vd))
        with open('iou_td.txt','a') as f:
            f.write('{}\n'.format(iou_td))
        with open('iou_vd.txt', 'a') as f:
            f.write('{}\n'.format(iou_vd))
        
        if mn is None:
            mn = 'fcn_{}.pth'.format(e)
            torch.save(net.state_dict(), mn)
        if temp > loss_td: ##############################注意啊这里被改成看训练数据的loss了
            os.remove(mn)
            mn = 'fcn_{}.pth'.format(e)
            torch.save(net.state_dict(), mn)
            temp = loss_td


    print(time.asctime())
