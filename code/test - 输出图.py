import os
import shutil
import time
import numpy as np
import torch
import gdal
import glob
from FCN import FCN

myseed = 810975  # set a random seed for reproducibility 随机种子
torch.backends.cudnn.deterministic = True # 固定CUDA随机数，这样相同超参数跑出来的网络都是一样的，得配合torch.manual_seed使用
#if benchmark=True, deterministic will be False
torch.backends.cudnn.benchmark = False # 为你精心选择最快的卷积层优化算法，我不需要谢谢
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available(): #如果你有CUDA
    torch.cuda.manual_seed_all(myseed)#为CUDA所有随机数添加种子

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
        (input_vertex, output_vertex, hidden_vertex, num_layer,BN,Drop) = net_para
        net = FCN(input_vertex, output_vertex, hidden_vertex, num_layer,BN,Drop).cuda()  #
        net.load_state_dict(torch.load(net_file)) #map_location=torch.device('cpu')
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

    def normalization(a):
        '''
        utc=00:00-00:50
        第1个波段的均值：0.17408108711242676，标准差：0.09180239588022232
        第2个波段的均值：0.15532344579696655，标准差：0.09411614388227463
        第3个波段的均值：0.13014985620975494，标准差：0.10350919514894485
        第4个波段的均值：0.17147856950759888，标准差：0.12056007981300354
        第5个波段的均值：0.10136411339044571，标准差：0.07353920489549637
        第6个波段的均值：0.06755834072828293，标准差：0.056784629821777344
        第7个波段的均值：291.5131530761719，标准差：12.34211540222168
        第8个波段的均值：235.845947265625，标准差：8.17346477508545
        第9个波段的均值：245.95281982421875，标准差：10.59644889831543
        第10个波段的均值：254.6614532470703，标准差：12.598808288574219
        第11个波段的均值：279.3176574707031，标准差：19.40409278869629
        第12个波段的均值：262.65631103515625，标准差：12.848060607910156
        第13个波段的均值：280.890380859375，标准差：20.239898681640625
        第14个波段的均值：279.4247741699219，标准差：20.82622718811035
        第15个波段的均值：276.036865234375，标准差：20.321340560913086
        第16个波段的均值：264.0698547363281，标准差：16.568769454956055
        '''
        Morning = [[0.17408109, 0.091802396], [0.15532345, 0.094116144], [0.13014986, 0.103509195],
                   [0.17147857, 0.12056008],
                   [0.10136411, 0.073539205], [0.06755834, 0.05678463], [291.51315, 12.342115], [235.84595, 8.173465],
                   [245.95282, 10.596449], [254.66145, 12.598808], [279.31766, 19.404093], [262.6563, 12.848061],
                   [280.89038, 20.239899], [279.42477, 20.826227], [276.03687, 20.32134], [264.06985, 16.56877],
                   [-0.81586134, 2.9236693], [2.6795657, 1.8712212]]
        Noon = [[0.24988365, 0.15231493], [0.22707175, 0.15462552], [0.19655935, 0.1661158], [0.27005583, 0.19260702],
                [0.1573529, 0.108633794], [0.106533796, 0.08361524], [294.75528, 12.182429], [235.45898, 7.0298758],
                [245.67133, 9.387297], [254.40428, 11.410765], [279.58807, 18.392067], [262.90427, 12.510294],
                [281.01477, 19.294218], [279.3004, 19.88638], [275.60718, 19.318005], [263.4919, 15.553772],
                [-0.40279675, 2.801445], [2.7492037, 1.7626182]]
        ary = np.copy(a)

        for i in range(ary.shape[1] - 3):
            mean = Morning[i][0]
            std = Morning[i][1]
            ary[:, i] = (ary[:, i] - mean) / std

        return ary

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
                # if 5 >= i >= 0:
                #     ary[:, i] = (ary[:, i] - 0) / (1.22 - 0)
                # if 15 >= i >= 6:
                #     ary[:, i] = (ary[:, i] - 180) / (401 - 180)
                if 17 >= i >= 16:
                    ary[:, i] = ary[:, i] / 10
        '''for i in range(ary.shape[1]):
            if 5 >= i >= 0:
                ary[:, i] = (ary[:, i] - 0.133) / 0.1 +1.1
            if 15 >= i >= 6:
                ary[:, i] = (ary[:, i] - 267.038) / 23.103 +1.1
            if 17 >= i >= 16:
                ary[:, i] = ary[:, i] - 0.932 / 3.013 +1.1'''



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
        tnr = TN/(TN+FP)
        return acc, pre, rcl,iou,fone,kpa,tnr

    def seive_data(ary, npz):
        #把图像中的训练数据根据npz提示的行列号给筛选出来,做成Data
        idx = []
        for i in range(501):
            for j in range(582):
                idx.append([i, j])
        data = []
        td_idx = np.load(npz)['td_idx']
        td_idx.tolist()
        zero = np.zeros([501,582])

        for [r,c] in td_idx:
            zero[r,c]=1

        for [row,col] in idx:
            if not zero[row,col] == 1:
                data.append(ary[:,row,col])
        Data = np.array(data)
        return Data

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

        def Normalization(a):
            ary = np.copy(a)
            for i in range(ary.shape[0] - 1):
                ary[i, :, :] = (ary[i, :, :] - ary[i, :, :].mean()) / ary[i, :, :].std()
            return ary
        #samples = .tif文件名包含路径
        file = gdal.Open(smp_pth)
        ary = file.ReadAsArray()
        print(ary.shape)
        ###################偷梁换柱#################
        utc = smp_pth.split('\\')[-1][1:3]
        number = int(utc)
        # number = 40 - number  # 越靠近中午的越小
        # ary[16] = np.ones(ary[16].shape) * number * 0.05
        # number = 40 - number  # 越靠近中午的越小
        # semi_ary[16] = np.ones(semi_ary[16].shape) * number * 0.05
        ary[16] = np.ones(ary[16].shape) * np.sin(np.pi * number / 80)
        ###################偷梁换柱#################
        #ary = Normalization(ary)
        model_type = net_file.split('\\')[-3]
        model_num = net_file.split('\\')[-2]
        date = smp_pth.split('\\')[-2]
        Time = smp_pth.split('\\')[-1][:-4]
        npz = r'E:\SmokeDetection\source\MLP_cirrus_test\{}\{}\{}\{}_idx.npz'.format(model_type,model_num,date,Time)

        if  0:#date =='0830':################################筛选代码
            Data = seive_data(ary, npz)
        else:
            Data = ary.reshape([19, ary.shape[1] * ary.shape[2]]).T #这里需要把原始数据变形成shape=【291582,19】

        #Input = normalize(Data)
        Input = normalization(Data) #归一化标准化在这调整
        #Input = Data
        Input = choose_bands(Input, bands) # (50600, 17)
        #print(Input[:,16])
        TP,TN,FP,FN=0,0,0,0
        Prediction = np.ones([Data.shape[0]])*255
        step = 7500
        for I in range(0, Data.shape[0], step):   #这里8G显存不能完整塞下一张图，只好拆一下了
            label = torch.from_numpy(Data[I:I + step, -1]).long().cuda()
            data = torch.from_numpy(Input[I:I + step]).float().cuda() #格式转换
            pred = net(data)
            label.detach()
            data.detach()
            pred = torch.softmax(pred,1)
            pred.detach()
            prediction = torch.max(pred, 1)[1]
            #prediction = torch.where(pred[:,1]>0.65,1,0)#阈值在这调整
            Prediction[I:I + step]=prediction.cpu().detach().numpy()
            tp, tn, fp, fn = cal_cfm(prediction.cpu(), label.cpu())
            TP+=tp
            TN+=tn
            FP+=fp
            FN+=fn
            del data, label, pred
        if 1:  # 成图代码
            img=Prediction.reshape([ary.shape[1], ary.shape[2]])*255
            mk_pred_pic(img, file,
                        'self-training_test{}_{}'.format(smp_pth.split('\\')[-2],smp_pth.split('\\')[-1]))  # 成图代码
            print(TP, TN, FP, FN)
        return TP, TN, FP, FN

    net = load_network()
    #sample = r'E:\SmokeDetection\source\new_data\samples0830_2020_10_28\08300000_0_192_nosmoke.tif'
    accuracy, precision, recall, f1, kappa, IoU, TNR = 0,0,0,0,0,0,0
    for count,sample in enumerate(test_samples_files):
        TP, TN, FP, FN = output_one_sample(sample, net)
        acc, pre, rcl, iou, fone, kpa,tnr = cal_evaluation(TP, TN, FP, FN)
        accuracy += acc
        precision += pre
        recall += rcl
        IoU += iou
        #f1 += fone
        kappa += kpa
        TNR += tnr
        #print('处理了{}幅图。'.format(i+1))
        string1 ='图片名称：{}_{}\n'.format(sample.split('\\')[-2],sample.split('\\')[-1])
        string2='评价参数：Accuracy = {}|Precision = {} | Recall = {}| IoU = {}| F1score = {}| Kappa coefficient={}| TNR = {}\n'.format(acc,pre,rcl,iou,fone,kpa,tnr)
        with open(result_pth + os.sep + 'Evaluation report_{}_{}.txt'.format(EveryPicName,name), 'a') as f:
            f.write(string1)
            f.write(string2)
            print(string1)
            print(string2)
    accuracy /= count+1
    precision /= count+1
    recall /= count+1
    IoU /= count+1
    f1 =2*precision*recall/(precision+recall)
    kappa /= count+1
    TNR /= count+1
    f2 = 2*TNR*recall/(TNR+recall)
    string = 'Accuracy = {}\nPrecision = {}\nRecall = {}\nIoU = {}\nF1score = {}\nKappa coefficient={}\nTNR={}\nf2={}'.format(accuracy,precision,recall,IoU,f1,kappa,TNR,f2)
    with open(result_pth+os.sep+'Evaluation report_{}_{}.txt'.format(AllTestName,name),'w') as f:
        f.write(string)
        print('成功输出评价参数结果')


EveryPicName = '0830_EveryMorningPicture'#'fourdaysNoTrain_EveryPicture' #'0830NoTrain_EveryPicture'
AllTestName = '0830'#'fourdays'#'0830'
#net_files = glob.glob(r'E:\SmokeDetection\source\MLP_90_exp\4000_smoke\1\*f1.pth')
net_files = glob.glob(r'E:\SmokeDetection\source\semi-supervised learning\result\test 0830 2hours with time semi065 small\*89.pth') #结果存储
for net_file in net_files:
    #filePath = r'E:\SmokeDetection\source\MLP_Results\256_3_nowind_200smoke1800nosmoke'
    #net_file = filePath + r'\fcn_interval_199.pth'
    filePath = net_file.split('\\f')[:-1][0]
    name = net_file.split('\\')[-2]+net_file.split('\\')[-1][:-4]
    bands =[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]#[1,2,3,7,11,13,14,15,16]
    #net_para =(input_vertex, output_vertex, hidden_vertex, num_layer, BN, Drop)
    net_para = (len(bands), 2, 17, 1,True,0) #超参数定义！！！！！！！！！！！！！！!!!!!!!!!!!!!!!!!!!!!!!!!!!!！！！！！！
    test_samples_files = glob.glob(r'E:\SmokeDetection\source\semi-supervised learning\cliped_new_new_data\0830\0*0.tif')[:6]
    #test_samples_files = glob.glob(r'E:\SmokeDetection\source\hist_data\*\*.tif')
    #test_samples_files = glob.glob(r'E:\SmokeDetection\source\test_data\*\*.tif')[6:]
    result_pth = filePath
    string3 = '正在运行模型{}'.format(net_file.split('source')[-1])
    with open(result_pth + os.sep + 'Evaluation report_{}_{}.txt'.format(EveryPicName,name), 'w') as f:
        f.write(string3)
        print(string3)
    test_samples(net_file,net_para,test_samples_files,result_pth,name,bands)
    print('跑完网络模型{}了'.format(name))
