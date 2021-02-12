import os
import shutil
import time
import numpy as np
import torch
import gdal
import glob
from FCN import FCN

def test_samples(net_file, net_para, test_samples_files, result_pth, name, bands):
    '''
    输入测试集路径，得出所有的样本的总精度、F1、IOU、precision、recall、kappa
    也可输入一张图的路径，得出这张图的prediction的结果，以及这张图的TP\TN\FP\FN
    net_file = 文件存储路径包括名字
    net_para = 网络参数
    test_samples_files = 测试集样本路径
    result_pth = 结果存放路径
    name = 评价参数结果存储名
    bands = 选择的葵花影像和风数据通道，葵花数据是1-16通道，风数据是17，18通道，标签是19通道
    '''
    def load_network():
        #加载网络，输出的是训练好的网络模型
        (input_vertex, output_vertex, hidden_vertex, num_layer) = net_para
        net = FCN(input_vertex, output_vertex, hidden_vertex, num_layer).cuda()  #
        net.load_state_dict(torch.load(net_file))
        net = net.eval()
        return net

    def choose_bands(data, bands):
        '''
        这是选择对应波段加入神经网络，
        bands是一个列表，里面的数字代表选取对应波段，使用的是人类语言，比如选择1，2，3波段输入网络就是bands=[1,2,3]
        '''
        output = np.copy(data[:, bands[0] - 1])  # shape=750
        for band in bands[1:]:
            output = np.c_[output, data[:, band - 1]]
        return output

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
                ary[:, i] = (ary[:, i] - 0) / (1.22 - 0)
            if 15 >= i >= 6:
                ary[:, i] = (ary[:, i] - 180) / (401 - 180)
            if 17 >= i >= 16:
                ary[:, i] = ary[:, i] / 10
        return ary

    def cal_evaluation(TP, TN, FP, FN):
        # 只需要输入混淆矩阵四个值，输出各种评价参数
        T = TP + TN + FP + FN
        acc = (TP + TN) / T
        pre = TP / (TP + FP +1e-8)
        rcl = TP / (TP + FN + 1e-8)
        iou = TP / (TP + FN + FP + 1e-8)
        fone = 2 * pre * rcl / (pre + rcl+1e-8)
        temp = ((TP + FP) * (TP + FN) + (TN + FP) * (TN + FN)) / (T * T)
        kpa = (acc - temp) / (1 - temp +1e-9)
        return acc, pre, rcl, iou, fone, kpa

    def output_one_sample(smp_pth, net):
        #输出一张图像的评价参数以及预测值
        def cal_cfm(pred_choice, target):
            # 计算一张图的混淆矩阵
            TP, TN, FP, FN = 0, 0, 0, 0
            # TP    predict 和 label 同时为1
            TP += ((pred_choice == 1) & (target.data == 1)).sum().float()
            # TN    predict 和 label 同时为0
            TN += ((pred_choice == 0) & (target.data == 0)).sum().float()
            # FN    predict 0 label 1
            FN += ((pred_choice == 0) & (target.data == 1)).sum().float()
            # FP    predict 1 label 0
            FP += ((pred_choice == 1) & (target.data == 0)).sum().float()
            return TP, TN, FP, FN

        def mk_pred_pic(Prediction, file, result_name):
            driver = gdal.GetDriverByName('GTiff')
            print(result_pth+os.sep+result_name)
            out = driver.Create(result_pth+os.sep+result_name, Prediction.shape[1], Prediction.shape[0])
            out.GetRasterBand(1).WriteArray(Prediction*255)
            out.SetGeoTransform(file.GetGeoTransform())
            out.SetProjection(file.GetProjection())
            print('成功输出预测图')
        #samples = .tif文件名包含路径
        file = gdal.Open(smp_pth)
        ary = file.ReadAsArray()
        Data = ary.reshape([19, ary.shape[1] * ary.shape[2]]).T #这里需要把原始数据变形成shape=【19,291582】
        Input = normalize(Data)
        Input = choose_bands(Input, bands)
        TP,TN,FP,FN=0,0,0,0
        Prediction = np.ones([291582])*255
        step = 7500
        for I in range(0, 291582, step):   #这里8G显存不能完整塞下一张图，只好拆一下了
            label = torch.from_numpy(Data[I:I + step, -1]).long().cuda()
            data = torch.from_numpy(Input[I:I + step]).float().cuda() #格式转换
            pred = net(data)
            label.detach()
            data.detach()
            pred = torch.softmax(pred,1)
            pred.detach()
            prediction = torch.max(pred, 1)[1]
            #prediction = pred[:,1]
            Prediction[I:I + step]=prediction.cpu().detach().numpy()
            tp, tn, fp, fn = cal_cfm(prediction.cpu(), label.cpu())
            TP+=tp
            TN+=tn
            FP+=fp
            FN+=fn
            del data, label, pred
        if 0:  # 成图代码
            img=Prediction.reshape([ary.shape[1], ary.shape[2]])
            mk_pred_pic(img, file,
                        'softest_{}_{}'.format(smp_pth.split('\\')[-2],smp_pth.split('\\')[-1]))  # 成图代码
            print(TP, TN, FP, FN)
        return TP, TN, FP, FN

    net = load_network()
    #sample = r'E:\SmokeDetection\source\new_data\samples0830_2020_10_28\08300000_0_192_nosmoke.tif'
    accuracy, precision, recall, f1, kappa, IoU = 0,0,0,0,0,0
    for i,sample in enumerate(test_samples_files):
        TP, TN, FP, FN = output_one_sample(sample, net)
        acc, pre, rcl, iou, fone, kpa = cal_evaluation(TP, TN, FP, FN)
        accuracy += acc
        precision += pre
        recall += rcl
        IoU += iou
        f1 += fone
        kappa += kpa
        print('处理了{}幅图。'.format(i+1))
    accuracy /= i+1
    precision /= i+1
    recall /= i+1
    IoU /= i+1
    f1 /= i+1
    kappa /= i+1
    string = 'Accuracy = {}\n Precision = {}\n Recall = {}\n IoU = {}\n F1score = {}\n Kappa coefficient={}'.format(accuracy,precision,recall,IoU,f1,kappa)
    with open(result_pth+os.sep+'Evaluation report_test data_{}_199 .txt'.format(name),'w') as f:
        f.write(string)
        print('成功输出评价参数结果')



#net_files = glob.glob(r'E:\SmokeDetection\source\MLP_Results\*\*.pth')
# #for net_file in net_files:
filePath = r'I:\MLP_RESULT\20210209\256_3_100smoke1900nosmoke'
net_file = filePath + r'\fcn_interval_199.pth'
name = net_file.split('\\')[-2]
bands = [1,2,3,13,14,15,16,17,18]
net_para = len(bands), 2, eval(name.split('_')[0]), eval(name.split('_')[1])
#test_samples_files = glob.glob(r'E:\SmokeDetection\source\new_new_data\0817\0010.tif')
#test_samples_files = glob.glob(r'E:\SmokeDetection\source\new_new_data\*\*.tif')
test_samples_files = glob.glob(r'E:\SmokeDetection\source\test_data\*\*.tif')
result_pth = filePath
test_samples(net_file,net_para,test_samples_files,result_pth,name,bands)
print('跑完网络模型{}了'.format(name))
