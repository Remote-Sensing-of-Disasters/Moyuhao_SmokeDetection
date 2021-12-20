from FCN import FCN
import torch
import os
import numpy as np
import glob
import gdal
import time
import torch.utils.data as Data
import random

myseed = 810975  # set a random seed for reproducibility 随机种子
torch.backends.cudnn.deterministic = True # 固定CUDA随机数，这样相同超参数跑出来的网络都是一样的，得配合torch.manual_seed使用
#if benchmark=True, deterministic will be False
torch.backends.cudnn.benchmark = False # 为你精心选择最快的卷积层优化算法，我不需要谢谢
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available(): #如果你有CUDA
    torch.cuda.manual_seed_all(myseed)#为CUDA所有随机数添加种子

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

     20210430统一短长波的均值方差法
     mean=0.13332568109035492,std=0.10010378062725067
    mean=267.0377502441406,std=23.10291862487793
    mean=0.9318513870239258,std=3.01316237449646
    '''
    ary = np.copy(a)
    '''for i in range(ary.shape[1]):
        if 5 >= i >= 0:
            ary[:, i] = (ary[:, i] - 0) / (1.22 - 0)
        if 15 >= i >= 6:
            ary[:, i] = (ary[:, i] - 180) / (401 - 180)
        if 17 >= i >= 16:
            ary[:, i] = ary[:, i] / 10'''

    '''
    for i in range(ary.shape[1]):
        if 5 >= i >= 0:
            ary[:,i] = (ary[:,i]-0.133)/0.1 +1.1
        if 15 >= i >= 6:
            ary[:,i] = (ary[:,i]-267.038)/23.103 +1.1
        if 17>= i >= 16:
            ary[:, i] = ary[:, i]-0.932 / 3.013 +1.1
    '''
    return ary

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

    utc=01:00--???? 20210529
    第1个波段的均值：0.24988365173339844，标准差：0.1523149311542511
    第2个波段的均值：0.22707174718379974，标准差：0.1546255201101303
    第3个波段的均值：0.1965593546628952，标准差：0.16611580550670624
    第4个波段的均值：0.2700558304786682，标准差：0.19260701537132263
    第5个波段的均值：0.15735289454460144，标准差：0.10863379389047623
    第6个波段的均值：0.10653379559516907，标准差：0.083615243434906
    第7个波段的均值：294.7552795410156，标准差：12.182429313659668
    第8个波段的均值：235.458984375，标准差：7.029875755310059
    第9个波段的均值：245.67132568359375，标准差：9.387296676635742
    第10个波段的均值：254.40428161621094，标准差：11.410764694213867
    第11个波段的均值：279.58807373046875，标准差：18.392066955566406
    第12个波段的均值：262.9042663574219，标准差：12.510293960571289
    第13个波段的均值：281.0147705078125，标准差：19.294218063354492
    第14个波段的均值：279.3004150390625，标准差：19.88637924194336
    第15个波段的均值：275.607177734375，标准差：19.318004608154297
    第16个波段的均值：263.4919128417969，标准差：15.55377197265625
    第17个波段的均值：-0.40279674530029297，标准差：2.8014450073242188
    第18个波段的均值：2.749203681945801，标准差：1.7626181840896606

    Process finished with exit code 0

    '''
    Morning =[[0.17408109, 0.091802396], [0.15532345, 0.094116144], [0.13014986, 0.103509195], [0.17147857, 0.12056008],
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

    for i in range(ary.shape[1]-3):
        mean = Morning[i][0]
        std = Morning[i][1]
        ary[:, i] = (ary[:, i] - mean) / std


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

def train(td_path, vd_path, result_path, bands, output_vertex, hidden_vertex, num_layer, echo, batchsize, lr, filesize,BN,Drop,net_file,semi):
    def load_network():
        # 加载网络，输出的是训练好的网络模型
        net = FCN(input_vertex, output_vertex, hidden_vertex, num_layer, BN, Drop).cuda()  #
        net.load_state_dict(torch.load(net_file))  # map_location=torch.device('cpu')
        return net

    def get_pseudo_labels(dataset, model, threshold=0.0):
        # This functions generates pseudo-labels of a dataset using given model.
        # It returns an instance of DatasetFolder containing images whose prediction confidences exceed a given threshold.
        # You are NOT allowed to use any models trained on external data for pseudo-labeling.
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Construct a data loader.
        data_loader = Data.DataLoader(dataset, batch_size=batchsize, shuffle=True)

        # Make sure the model is in eval mode.
        model.eval()
        # Define softmax function.
        softmax = torch.nn.Softmax(dim=-1)
        argsmp = []
        pseudo_label = []
        # Iterate over the dataset by batches.
        for batch in (data_loader):
            img, _ = batch # img.size()=128*17 batch*channel
            # Forward the data
            # Using torch.no_grad() accelerates the forward process.
            with torch.no_grad():
                logits = model(img.to(device))
            # Obtain the probability distributions by applying softmax on logits.
            probs = softmax(logits.detach())

            # ---------- TODO ----------
            # Filter the data and construct a new dataset.
            info0, info1 = torch.max(probs, dim=1)[0], torch.max(probs, dim=1)[1]
            smoke_num, nosmoke_num = 0,0 # 这是初始的有烟像元限定数和无烟限定数
            for j, (prob, label) in enumerate(zip(info0, info1)):
                if prob >= threshold:
                    if label==1:
                        argsmp.append(img[j])
                        pseudo_label.append(label.long())
                        smoke_num += 1 # 当为烟像元的时候，就把smoke_num + 1
                        nosmoke_num += 49 # 暂时限定无烟的数量最多是有烟的49倍,根据0100的有烟占比0.017搞出来的
                    if label==0 and nosmoke_num>0 :
                        argsmp.append(img[j])
                        pseudo_label.append(label.long())
                        nosmoke_num -= 1
        argsmp = torch.stack(argsmp).cuda()
        pseudo_label = torch.stack(pseudo_label).cuda()  # 图片太多Concat的时候会溢出
        with open('pseudo_samples.txt', 'a') as f:
            print('有{}个半监督样本，其中有烟的有{}个\n'.format(pseudo_label.size()[0],pseudo_label.sum()))
            f.write('有{}个半监督样本，其中有烟的有{}个\n'.format(pseudo_label.size()[0],pseudo_label.sum()))
        dataset = Data.TensorDataset(argsmp, pseudo_label)
        if len(argsmp) == 0:
            return 0
        # # Turn off the eval mode.
        model.train()
        del argsmp, pseudo_label
        return dataset

    #bands指的是输入的通道序号，葵花卫星短到长波为从1到16，风数据从17到18，标签数据19（-1）
    print(time.asctime())
    input_vertex = len(bands)
    os.chdir(result_path)
    fns = td_path  # 接口改在外面了
    mn = None
    mn1= None
    temp = 1000
    temp2=0
    #加载网络和优化器
    net = FCN(input_vertex, output_vertex, hidden_vertex, num_layer, BN, Drop).cuda()
    if net_file: net = load_network() #微调代码
    opt = torch.optim.Adam(params=net.parameters(),lr=lr) #BatchSize=2000+
    #opt = torch.optim.SGD(params=net.parameters(),momentum=0.9,lr=lr)#weight_decay=L2正则
    #加载训练数据
    tdShapeSize = int(filesize*0.7)
    trainingDataNumpy = np.zeros([tdShapeSize * len(fns), 19])
    for file_num, file in enumerate(fns):
        npfile = np.load(file) ###################偷梁换柱#################
        utc = file.split('\\')[-1][1:3]
        number = int(utc)
        #number = 40-number # 越靠近中午的越小
        #npfile[:,16] = np.ones(npfile[:,16].shape)*number*0.05 #这个0.05为了缩小距离，减少一点稀疏问题，17波段当作时间变量
        npfile[:,16] = np.ones(npfile[:,16].shape)*np.sin(np.pi * number / 80)
        trainingDataNumpy[(file_num) * tdShapeSize:(file_num + 1) * tdShapeSize, :] = npfile
    trainingTargetTorch = torch.from_numpy(trainingDataNumpy[:,-1]).cuda()
    #trainingDataNumpy = normalize(trainingDataNumpy) 原来在1月份，我的封装就已经达到了准工业级的境界(吹的）
    trainingDataNumpy = normalization(trainingDataNumpy) # (425040, 19)
    trainingDataNumpyNor = choose_bands(trainingDataNumpy,bands)
    trainingDataTorch = torch.from_numpy(trainingDataNumpyNor).cuda()
    #加载验证数据
    vdShapesize =int(filesize*0.3)
    fns = vd_path
    validationDataNumpy = np.zeros([vdShapesize * len(fns), 19])
    for file_num, file in enumerate(vd_path):
        npfile = np.load(file)  ###################偷梁换柱#################
        utc = file.split('\\')[-1][1:3]
        number = int(utc)
        # number = 40 - number  # 越靠近中午的越小
        # npfile[:,16] = np.ones(npfile[:,16].shape)*number*0.05 # 这个0.05为了缩小距离，减少一点稀疏问题，17波段当作时间变量
        npfile[:, 16] = np.ones(npfile[:, 16].shape) * np.sin(np.pi * number / 80)
        validationDataNumpy[(file_num) * vdShapesize:(file_num + 1) * vdShapesize, :] = npfile
    validationTargetTorch = torch.from_numpy(validationDataNumpy[:, -1]).cuda()
    #validationDataNumpy = normalize(validationDataNumpy)
    validationDataNumpy = normalization(validationDataNumpy) # (425040, 19)
    validationDataNumpyNor = choose_bands(validationDataNumpy, bands)
    validationDataTorch = torch.from_numpy(validationDataNumpyNor).cuda()
    #print(validationDataTorch.size(),validationTargetTorch.size())

    # 加载要制作假标签的数据
    if semi :
        import gdal
        reshape_target = []
        reshape_data = []
        for fn in semi:
            semi_ary = gdal.Open(fn).ReadAsArray() # shape = (19, 230, 220)
            ###################偷梁换柱#################
            utc = fn.split('\\')[-1][1:3]
            number = int(utc) # 0400=40
            # number = 40 - number  # 越靠近中午的越小
            # semi_ary[16] = np.ones(semi_ary[16].shape) * number * 0.05
            semi_ary[16] = np.ones(semi_ary[16].shape) * np.sin(np.pi*number/80)
            ###################偷梁换柱#################
            trans_ary = semi_ary.transpose()
            reshape_ary = trans_ary.reshape(trans_ary.shape[0]*trans_ary.shape[1],trans_ary.shape[2]) # shape = (50600, 19)
            pseudoTargetTorch = torch.from_numpy(reshape_ary[:, -1]).cuda() # size = (50600, 1) 这里一定要把标签带上因为我懒得改了
            reshape_target.append(pseudoTargetTorch)
            pseudoDataNumpy = normalization(reshape_ary)
            pseudoDataNumpyNor = choose_bands(pseudoDataNumpy, bands)
            pseudoDataTorch = torch.from_numpy(pseudoDataNumpyNor).cuda()
            reshape_data.append(pseudoDataTorch)
        pseudoData = torch.cat(reshape_data,0)
        pseudoTarget= torch.cat(reshape_target,0)
        pseudoDataset = Data.TensorDataset(pseudoData.float(), pseudoTarget.long())

    print('数据组织完毕')

    for e in range(echo):
        trainingDataset = Data.TensorDataset(trainingDataTorch.float(), trainingTargetTorch.long())
        validationDataset = Data.TensorDataset(validationDataTorch.float(), validationTargetTorch.long())
        if semi and e>=0: #20211109 ##################################semi 半监督#############################
            semiDataset = get_pseudo_labels(pseudoDataset, net, 0.6)
            if semiDataset: #20211109
                trainingDataset = Data.ConcatDataset([trainingDataset, semiDataset])
                validationDataset = Data.ConcatDataset([validationDataset, semiDataset])

        trainingDataLoader = Data.DataLoader(dataset=trainingDataset, batch_size=batchsize, shuffle=True)
        validationDataLoader = Data.DataLoader(dataset=validationDataset, batch_size=batchsize, shuffle=True)
        loss_td = 0
        loss_vd = 0

        tp, tn, fp, fn = 0, 0, 0, 0
        TP, TN, FP, FN = 0, 0, 0, 0

        #random.shuffle(fns)
        #fns = [r'E:\SmokeDetection\source\new_samples_64\0823\0000_1344_1088_smoke.tif'] ##!!!
        td_back_time = 0#每个周期的反向传播次数
        net.train()
        for step,(batchData, batchTarget) in enumerate(trainingDataLoader):
            opt.zero_grad()
            trainingPrediction = net(batchData)
            trainingLoss = torch.nn.CrossEntropyLoss()(trainingPrediction, batchTarget)
            #trainingLoss = torch.nn.MSELoss()(trainingPrediction, batchTarget.float())
            trainingLoss.backward()
            opt.step()
            trainingLoss.detach()
            trainingPrediction.detach()
            #print(trainingLoss)
            loss_td+=trainingLoss
            prediction = torch.max(trainingPrediction, 1)[1].cpu().numpy()
            gt = batchTarget.cpu().numpy()
            td_back_time +=1
            #print(step)
            for i in range(gt.shape[0]):
                if gt[i] == 1:
                    if prediction[i] == 1:
                        tp += 1
                    if prediction[i] == 0:
                        fn += 1
                        # find_pix_index(ary1[i,:], file)
                elif gt[i] == 0:
                    if prediction[i] == 0:
                        tn += 1
                    if prediction[i] == 1:
                        fp += 1
                        # find_pix_index(ary1[i,:], file)
            del prediction,trainingLoss
            loss_td=loss_td.detach()
        accuracy_td, precision_td, recall_td, iou_td, fone_td, kpa_td = cal_evaluation(tp, tn, fp, fn)
        loss_td /= td_back_time
        print('echo={},finish train time={}'.format(e, time.asctime()))
        os.chdir(result_path)
        print('Training precision：{}'.format(precision_td))
        print('Training recall：{}'.format(recall_td))
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
        del loss_td, trainingPrediction
        # ------------------------------------------验证数据---------------------------------------------#
        val_fns = vd_path
        vd_back_time = 0
        net.eval()
        for step, (batchData, batchTarget) in enumerate(validationDataLoader):
            with torch.no_grad():
                validationPrediction = net(batchData)
                validationLoss = torch.nn.CrossEntropyLoss()(validationPrediction, batchTarget)
                #validationLoss = torch.nn.MSELoss()(validationPrediction, batchTarget.float())
                validationLoss.detach()
                validationPrediction.detach()
                # print(trainingLoss)
                loss_vd += validationLoss
                #print('验证损失：{}'.format(validationLoss))
                prediction = torch.max(validationPrediction, 1)[1].cpu().numpy()
                gt = batchTarget.cpu().numpy()
            vd_back_time += 1
            # print(step)
            for i in range(gt.shape[0]):
                if gt[i] == 1:
                    if prediction[i] == 1:
                        TP += 1
                    if prediction[i] == 0:
                        FN += 1
                        # find_pix_index(ary1[i,:], file)
                elif gt[i] == 0:
                    if prediction[i] == 0:
                        TN += 1
                    if prediction[i] == 1:
                        FP += 1
                        # find_pix_index(ary1[i,:], file)
            loss_vd=loss_vd.detach()
            del prediction
        accuracy_vd, precision_vd, recall_vd, iou_vd, fone_vd, kpa_vd = cal_evaluation(TP, TN, FP, FN)
        print('echo={},finish validation time={}'.format(e, time.asctime()))
        os.chdir(result_path)
        loss_vd /= vd_back_time
        with open('accuracy_vd.txt', 'a') as f:
            f.write('{}\n'.format(accuracy_vd))
        with open('loss_vd.txt', 'a') as f:
            f.write('{}\n'.format(loss_vd))
        with open('recall_vd.txt', 'a') as f:
            f.write('{}\n'.format(recall_vd))
        with open('precision_vd.txt', 'a') as f:
            f.write('{}\n'.format(precision_vd))
        with open('fone_vd.txt', 'a') as f:
            f.write('{}\n'.format(fone_vd))
        with open('iou_vd.txt', 'a') as f:
            f.write('{}\n'.format(iou_vd))
        with open('kappa_vd.txt', 'a') as f:
            f.write('{}\n'.format(kpa_vd))
        if mn is None:
            mn = 'fcn_{}loss.pth'.format(e)
            mn1 = 'fcn_{}f1.pth'.format(e)
            torch.save(net.state_dict(), mn)
            torch.save(net.state_dict(), mn1)
        if temp2 < fone_vd:  ##############################注意啊这里被改成看验证数据的f1了
            os.remove(mn1)
            mn1 = 'fcn_{}_f1.pth'.format(e)
            torch.save(net.state_dict(), mn1)
            temp2 = fone_vd
        if temp > loss_vd:  ##############################注意啊这里被改成看验证数据的loss了
            os.remove(mn)
            mn = 'fcn_{}_loss.pth'.format(e)
            torch.save(net.state_dict(), mn)
            temp = loss_vd
        print(time.asctime())
        del loss_vd,validationLoss,validationPrediction
        if e % 20 == 19:
            saveInterval = 'fcn_interval_{}.pth'.format(e)
            torch.save(net.state_dict(), saveInterval)
'''  for count,file in enumerate(fns[file_num:file_num+filesize]):
                ary1 = np.load(file)
                ary = normalize(ary1)
                np.random.shuffle(ary)
                #ary = np.random.random(ary.shape) #造假数据看看
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
'''


