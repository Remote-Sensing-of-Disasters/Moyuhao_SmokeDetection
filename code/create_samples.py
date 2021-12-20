import os
import glob
import random
import gdal
import numpy as np

idx=[]
for i in range(230):
    for j in range(220):
        idx.append([i,j])

def Normalization(a):
    ary = np.copy(a)
    for i in range(ary.shape[0]-1):
        ary[i,:,:] = (ary[i,:,:]-ary[i,:,:].mean())/ary[i,:,:].std()
    return ary
def create_by_random_noise(file, td_pth, vd_pth, idx_pth, file_nexttime, alpha): #,label
    # 20211217 15:55 使用随机采样的方式加入噪声，采样的规则是：对最靠近待预测的时段的样本的每个像元的每个波段的值进行一个定向的噪声。
    # V1'=V1+α(V2-V1)
    # file_nexttime是待预测的图像
    # alpha就是公式里的α
    ary = gdal.Open(file).ReadAsArray()
    ary_temp =gdal.Open(file).ReadAsArray()
    # ary = Normalization(ary)
    #l = gdal.Open(label).ReadAsArray() / 255.0
    # l[:250,:] = l[:250,:] *0.0
    #ary[:-1, :, :] = l
    random.shuffle(idx)
    td = []
    vd = []
    td_idx = []
    vd_idx =[]
    count = 0 #计数君

    while len(td)<ary.shape[1]*ary.shape[2]*0.7: # 每幅图35420个像元用作训练，占一幅图的70%，总数50600 20210913
        td.append(ary[:,idx[count][0],idx[count][1]]+alpha*(ary_temp[:,idx[count][0],idx[count][1]]-ary[:,idx[count][0],idx[count][1]])) #读入数据
        td_idx.append(idx[count])
        count+=1

    while len(vd)<ary.shape[1]*ary.shape[2]*0.3: # 每幅图15180个像元用作验证，占一幅图的30% 一幅图shape=(19, 230, 220) 20210913
        vd.append(ary[:,idx[count][0],idx[count][1]]+alpha*(ary_temp[:,idx[count][0],idx[count][1]]-ary[:,idx[count][0],idx[count][1]]))
        vd_idx.append(idx[count])
        count+=1

    np.save(td_pth, td)
    np.save(vd_pth, vd)
    np.savez(idx_pth, td_idx=td_idx, vd_idx=vd_idx)

def create_by_random(file, td_pth, vd_pth, idx_pth): #,label
    ary = gdal.Open(file).ReadAsArray()
    # ary = Normalization(ary)
    #l = gdal.Open(label).ReadAsArray() / 255.0
    # l[:250,:] = l[:250,:] *0.0
    #ary[:-1, :, :] = l
    random.shuffle(idx)
    td = []
    vd = []
    td_idx = []
    vd_idx =[]
    count = 0 #计数君

    while len(td)<ary.shape[1]*ary.shape[2]*0.7: # 每幅图35420个像元用作训练，占一幅图的70%，总数50600 20210913
        td.append(ary[:,idx[count][0],idx[count][1]]) #读入数据
        td_idx.append(idx[count])
        count+=1

    while len(vd)<ary.shape[1]*ary.shape[2]*0.3: # 每幅图15180个像元用作验证，占一幅图的30% 一幅图shape=(19, 230, 220) 20210913
        vd.append(ary[:,idx[count][0],idx[count][1]])
        vd_idx.append(idx[count])
        count+=1

    np.save(td_pth, td)
    np.save(vd_pth, vd)
    np.savez(idx_pth, td_idx=td_idx, vd_idx=vd_idx)

def create_by_cirrus(file, label, td_pth, vd_pth,idx_pth, num):
    #2021.08.03加入半监督假标签
    ary = gdal.Open(file).ReadAsArray()
    l = gdal.Open(label).ReadAsArray()
    ary[:-1,:,:]=l
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


def create_by_smoke(file, label, td_pth, vd_pth, idx_pth, num):
    # 输入一张图片的文件名，Ground Truth文件名，训练数据的索引npy文件名，验证数据的索引npy文件名，每幅图采样的像元个数
    # 2021.08.03加入半监督假标签
    ''' if os.path.exists(td_pth):
        os.remove(td_pth)
        print('已经删除原来的训练集列表')
    if os.path.exists(vd_pth):
        os.remove(vd_pth)
        print('已经删除原来的验证集列表')
    #这两个if表示，如果提前存在就删除'''
    ary = gdal.Open(file).ReadAsArray()
    l = gdal.Open(label).ReadAsArray()/255.0
    #l[:250,:] = l[:250,:] *0.0
    ary[:-1,:,:] = l
    random.shuffle(idx)
    td = []
    vd = []
    td_idx = []
    vd_idx = []
    #把这些固定比例看作指标
    count = 0 #count被架空了，它没有了num引导的上限
    td_smoke_num = 2500 #num//20
    td_other_num = num-td_smoke_num
    vd_smoke_num = 1250
    vd_other_num = num//2-vd_smoke_num
    while td_smoke_num+td_other_num+vd_smoke_num + vd_other_num>0:  # 当这两个指标都没用完就继续跑
        if count == (ary.shape[1]*ary.shape[2]):
            random.shuffle(idx)
            count=0
            print('干了一轮，训练烟像元还差{}个'.format(td_smoke_num))
            print('干了一轮，验证烟像元还差{}个'.format(vd_smoke_num))
            continue
        if ary[-1, idx[count][0], idx[count][1]] == 1 and td_smoke_num>0: #烟像元判定
            td.append(ary[:, idx[count][0], idx[count][1]])  # 读入数据
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
        if not ary[-1, idx[count][0], idx[count][1]] == 1 and td_other_num > 0:
            td.append(ary[:, idx[count][0], idx[count][1]])  # 读入数据
            td_idx.append(idx[count])  # 读入索引
            count += 1
            td_other_num -= 1
            continue
        if ary[-1, idx[count][0], idx[count][1]] == 1 and vd_smoke_num>0:
            vd.append(ary[:, idx[count][0], idx[count][1]])
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
        if not ary[-1, idx[count][0], idx[count][1]] == 1 and vd_other_num > 0:
            vd.append(ary[:, idx[count][0], idx[count][1]])  # 读入数据
            vd_idx.append(idx[count])  # 读入索引
            count += 1
            vd_other_num -= 1
            continue
        count+=1

    np.save(td_pth, td)
    np.save(vd_pth, vd)
    np.savez(idx_pth, td_idx=td_idx, vd_idx=vd_idx)



'''
fn = r'E:\SmokeDetection\source\semi-supervised learning\psuedo_samples\0050-0140'#数据存放点!!!!记住了，一定要把上个时刻的数据丢到这个文件夹里
utc = '0050'#数据的时间   记得确认数据哦
result =r'noon_model' #上一个模型跑出来的label，正类255，负类0
pic = r'E:\SmokeDetection\source\semi-supervised learning\cliped_new_new_data\0830\{}.tif'.format(utc)
lbl = r'E:\SmokeDetection\source\semi-supervised learning\result\{}\self-training_test0830_{}.tif'.format(result, utc)
create_by_random(pic, fn+r'\{}_td.npy'.format(utc),
                    fn+r'\{}_vd.npy'.format(utc),
                     fn+r'\{}_idx.npz'.format(utc),lbl)

'''

fns = glob.glob(r'E:\SmokeDetection\source\semi-supervised learning\cliped_new_new_data\*\0100.tif')
for f in fns:
    date = f.split('\\')[-2]
    fn = r'E:\SmokeDetection\source\semi-supervised learning\manual_samples\{}'.format(date)#数据存放点!!!!记住了，一定要把上个时刻的数据丢到这个文件夹里
    utc = f[-8:-4]#数据的时间   记得确认数据哦
    pic = f
    create_by_random_noise(pic, fn+r'\{}_01_td.npy'.format(utc),
                    fn+r'\{}_01_vd.npy'.format(utc),
                     fn+r'\{}_01_idx.npz'.format(utc),r'E:\SmokeDetection\source\semi-supervised learning\cliped_new_new_data\{}\0050.tif'.format(date),0.1)
    print('搞定一张图,date={},utc={}'.format(date,utc))
