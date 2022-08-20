import os
import glob
import random
from osgeo import gdal
import numpy as np

idx=[]
for i in range(501):
    for j in range(582):
        idx.append([i,j])
def noCloudNor(ary):
    smoke = []
    for i in range(ary.shape[1]):
        for j in range(ary.shape[2]):
            if ary[0, i, j] < 0.2: # 采样阈值判定
                smoke.append(ary[:, i, j])
    smoke = np.array(smoke) #shape=(288364, 19)08170000
    temp = np.copy(ary)
    for band in range(ary.shape[0]-1):
        std = smoke[:,band].std()
        mean = smoke[:,band].mean()
        temp[band,:,:]=(ary[band,:,:]-mean)/(std+1e-8)
    #print(temp.min())
    return temp

def instance_normalization(ary):
    '''
    对单个图片进行每个波段的归一化，就是Instance normalization，假设第一个维度是channel
    '''
    temp = np.copy(ary)
    for i in range(ary.shape[0]-3):
        if 1: #均值标准差
            mean = ary[i,:,:].mean()
            std = ary[i,:,:].std()
            temp[i,:,:] = (ary[i,:,:]-mean)/(std+1e-9)
        if 0: # 最大最小值
            Max = ary[i, :, :].max()
            Min = ary[i, :, :].min()
            temp[i, :, :] = (ary[i, :, :] - Min) / ((Max-Min) + 1e-9)
    return temp
def equalizeHist(img):
    #直方图均衡化
    for i in range(img.shape[0]-2):
        ary=np.copy(img[i,:,:])
        p=[]
        for j in range(256):
            p.append(np.where(ary<=j,1,0).sum()/(ary.shape[0]*ary.shape[1]))
        for j in range(255,0, -1):
            ary=np.where(ary==j,np.uint8(p[j]*255+0.5),ary)
        img[i,:,:]=ary
    return img

def create_by_random(file, td_pth, vd_pth, idx_pth, num):
    ary = gdal.Open(file).ReadAsArray()

    random.shuffle(idx)
    td = []
    vd = []
    td_idx = []
    vd_idx =[]
    count = 0 #计数君

    while len(td)<num-2000: # 每幅图num个像元用作训练
        td.append(ary[:,idx[count][0],idx[count][1]]) #读入数据
        td_idx.append(idx[count])
        count+=1

    while len(td)<num:
        if 0.2 < ary[0, idx[count][0], idx[count][1]] < 0.24 :
            td.append(ary[:, idx[count][0], idx[count][1]])  # 读入数据
            td_idx.append(idx[count])  # 读入索引
        count += 1

    while len(vd)<num//2-1000: # 每幅图num//2个像元用作验证
        vd.append(ary[:,idx[count][0],idx[count][1]])
        vd_idx.append(idx[count])
        count+=1

    while len(vd)<num//2: #2021.4.16孤注一掷
        if 0.2 < ary[0, idx[count][0], idx[count][1]] < 0.24 :
            vd.append(ary[:, idx[count][0], idx[count][1]])  # 读入数据
            vd_idx.append(idx[count])  # 读入索引
        count += 1

    np.save(td_pth, td)
    np.save(vd_pth, vd)
    np.savez(idx_pth, td_idx=td_idx, vd_idx=vd_idx)

def create_by_cirrus(file, td_pth, vd_pth,idx_pth, num):
    ary = gdal.Open(file).ReadAsArray()

    random.shuffle(idx)
    td = []
    vd = []
    td_idx = []
    vd_idx = []
    # 把这些固定比例看作指标
    count = 0  # count被架空了，它没有了num引导的上限
    td_smoke_num = num // 20
    td_cloud_num = num//20
    td_other_num = num // 20 * 18
    vd_smoke_num = num // 40
    vd_cloud_num = num//40
    vd_other_num = num // 40 * 18
    if ary[-1,:,:].sum()<vd_smoke_num*3:
        vd_smoke_num = ary[-1,:,:].sum()-num//20
        vd_cloud_num = num/2 - vd_smoke_num - vd_other_num
    while td_smoke_num + td_other_num + td_cloud_num+vd_smoke_num + vd_other_num +vd_cloud_num> 0:  # 当这指标都没用完就继续跑
        if ary[-1, idx[count][0], idx[count][1]] == 1 and td_smoke_num > 0:  # 烟像元判定
            td.append(ary[:, idx[count][0], idx[count][1]])  # 读入数据
            td_idx.append(idx[count])  # 读入索引
            count += 1
            td_smoke_num -= 1
            continue
        else:
            if 0.2<ary[0, idx[count][0], idx[count][1]]<0.24 and td_cloud_num > 0:
                td.append(ary[:, idx[count][0], idx[count][1]])  # 读入数据
                td_idx.append(idx[count])  # 读入索引
                count += 1
                td_cloud_num -= 1
                continue
            elif td_other_num > 0:
                td.append(ary[:, idx[count][0], idx[count][1]])  # 读入数据
                td_idx.append(idx[count])  # 读入索引
                count += 1
                td_other_num -= 1
                continue
        if ary[-1, idx[count][0], idx[count][1]] == 1 and vd_smoke_num > 0:  # 烟像元判定
            vd.append(ary[:, idx[count][0], idx[count][1]])  # 读入数据
            vd_idx.append(idx[count])  # 读入索引
            count += 1
            vd_smoke_num -= 1
            continue
        else:
            if 0.2 < ary[0, idx[count][0], idx[count][1]] < 0.24 and vd_cloud_num > 0:
                vd.append(ary[:, idx[count][0], idx[count][1]])  # 读入数据
                vd_idx.append(idx[count])  # 读入索引
                count += 1
                vd_cloud_num -= 1
                continue
            elif vd_other_num > 0:
                vd.append(ary[:, idx[count][0], idx[count][1]])  # 读入数据
                vd_idx.append(idx[count])  # 读入索引
                count += 1
                vd_other_num -= 1
                continue
        count+=1

    np.save(td_pth, td)
    np.save(vd_pth, vd)
    np.savez(idx_pth, td_idx=td_idx, vd_idx=vd_idx)


def create_by_smoke(file, td_pth, vd_pth, idx_pth, num):
    # 输入一张图片的文件名，训练数据的索引npy文件名，验证数据的索引npy文件名，每幅图采样的像元个数
    ''' if os.path.exists(td_pth):
        os.remove(td_pth)
        print('已经删除原来的训练集列表')
    if os.path.exists(vd_pth):
        os.remove(vd_pth)
        print('已经删除原来的验证集列表')
    #这两个if表示，如果提前存在就删除'''
    def time_series(ary, file):
        '''
        20220622
            给17波段写上时间编码，原则是从0817当地时（东8时）0点起算，每天间隔1/30，每小时间隔1/30 * 1/24， 每十分钟间隔1/30 * 1/24 * 1/6
        '''
        month = float(file.split('\\')[-2][-3])
        day = float(file.split('\\')[-2][-2:])
        hour = float(file.split('\\')[-1][:2])
        minute = float(file.split('\\')[-1][2])
        interval_day = 1/30
        interval_hour = (1/30) * (1/24)
        interval_min = (1/30) * (1/24) * (1/6)
        if month ==9:
            time_num = interval_day*(14+day)+interval_hour*(hour+8)+interval_min*(minute)
        if month == 8:
            time_num = interval_day * (day-17) + interval_hour * (hour + 8) + interval_min * (minute)
        ary[16,:,:] = time_num*np.ones(ary[16,:,:].shape).astype('float32')
        return ary
    ary = gdal.Open(file).ReadAsArray()
    # ary = equalizeHist(ary) # 在最大最小化之前做一个直方图均衡化
    # ary = instance_normalization(ary) # 普通的Instance Nor
    ary1 = noCloudNor(ary)  ###########################################20220805
    #ary = noCloudNor(ary) #不普通的Instance Nor，去除了B1>0.2的云
    #ary = time_series(ary,file)
    random.shuffle(idx)
    td = []
    vd = []
    td_idx = []
    vd_idx = []
    #把这些固定比例看作指标
    count = 0 #count被架空了，它没有了num引导的上限
    td_smoke_num = 5000 #num//20
    td_other_num = num-td_smoke_num
    vd_smoke_num = 2500
    vd_other_num = num//2-vd_smoke_num
    while td_smoke_num+td_other_num+vd_smoke_num + vd_other_num>0:  # 当这两个指标都没用完就继续跑
        if count == (ary.shape[1]*ary.shape[2]):
            random.shuffle(idx)
            count=0
            print('干了一轮，训练烟像元还差{}个'.format(td_smoke_num))
            print('干了一轮，验证烟像元还差{}个'.format(vd_smoke_num))
            print('干了一轮，训练无烟像元还差{}个'.format(td_other_num))
            print('干了一轮，验证无烟像元还差{}个'.format(vd_other_num))
            print('---'*4)
            continue
        if ary[-1, idx[count][0], idx[count][1]] == 1 and td_smoke_num>0: #烟像元判定
            td.append(ary1[:, idx[count][0], idx[count][1]])  # 读入数据
            td_idx.append(idx[count]) #读入索引
            count += 1
            td_smoke_num -=1
            continue
        # 这里开始魔改疫苗 第2批疫苗band7<300,band14>280
        #第三批疫苗band7<280,band14<260
        #第四批疫苗band1=[0.15,0.3];band14=[280,300]
        #第五批疫苗band1=[0.05,0.2];band14=[290,300]
        # if 0.05<ary[0, idx[count][0], idx[count][1]] <0.2 and 290<ary[13, idx[count][0], idx[count][1]] <300 and td_cirrus_num>0:
        #     td.append(ary[:, idx[count][0], idx[count][1]])  # 读入数据
        #     td_idx.append(idx[count])  # 读入索引
        #     count += 1
        #     td_cirrus_num-=1
        #     continue
        if ary[-1, idx[count][0], idx[count][1]] == 0 and td_other_num > 0 : # 无烟像元判定（训练） and ary[0, idx[count][0], idx[count][1]] <0.2
            td.append(ary1[:, idx[count][0], idx[count][1]])  # 读入数据
            td_idx.append(idx[count])  # 读入索引
            count += 1
            td_other_num -= 1
            continue
        if ary[-1, idx[count][0], idx[count][1]] == 1 and vd_smoke_num>0:
            vd.append(ary1[:, idx[count][0], idx[count][1]])
            vd_idx.append(idx[count])  # 读入索引
            count += 1
            vd_smoke_num -= 1
            continue
        # if 0.05<ary[0, idx[count][0], idx[count][1]] <0.2 and 290<ary[13, idx[count][0], idx[count][1]] <300 and vd_cirrus_num>0:# 这里开始魔改疫苗
        #     vd.append(ary[:, idx[count][0], idx[count][1]])  # 读入数据
        #     vd_idx.append(idx[count])  # 读入索引
        #     count+=1
        #     vd_cirrus_num-=1
        #     continue
        if ary[-1, idx[count][0], idx[count][1]] == 0 and vd_other_num > 0 : #and ary[0, idx[count][0], idx[count][1]] <0.2
            vd.append(ary1[:, idx[count][0], idx[count][1]])  # 读入数据
            vd_idx.append(idx[count])  # 读入索引
            count += 1
            vd_other_num -= 1
            continue
        count+=1

    np.save(td_pth, td)
    np.save(vd_pth, vd)
    np.savez(idx_pth, td_idx=td_idx, vd_idx=vd_idx)




fns = glob.glob(r'E:\SmokeDetection\source\MLP_cirrus_test\50000_smoke2500\Morning_20220811_2')
for fn in fns:
    for date in ['0818','0817', '0821', '0823', '0825']:  #['0817', '0821', '0823', '0825'] ['0830'] #['0818','0905','0901','0910','0817', '0821', '0823', '0825','0830']
        pp = glob.glob(r'E:\SmokeDetection\source\new_new_data\{}\00*0.tif'.format(date)) +glob.glob(r'E:\SmokeDetection\source\new_new_data\{}\0100.tif'.format(date))
        for p in pp:
            create_by_smoke(p, fn+r'\{}\{}_td.npy'.format(date, p[-8:-4]),
                            fn+r'\{}\{}_vd.npy'.format(date, p[-8:-4]),
                             fn+r'\{}\{}_idx.npz'.format(date, p[-8:-4]),
                             eval(fn.split('\\')[-2].split('_')[-2]))
            print('搞定一张图')
        print('搞定一天')
    print('搞定{}文件夹'.format(fn))




'''if not os.path.exists(fn+r'\{}'.format(date)):
    os.mkdir(fn+r'\{}'.format(date))
    print(fn+r'\{}'.format(date))'''
