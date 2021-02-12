import os
import glob
import random
import gdal
import numpy as np
def create_new_index(file, td_pth, vd_pth):
    #输入一张图片的文件名，训练数据的索引npy文件名，验证数据的索引npy文件名
    ''' if os.path.exists(td_pth):
        os.remove(td_pth)
        print('已经删除原来的训练集列表')
    if os.path.exists(vd_pth):
        os.remove(vd_pth)
        print('已经删除原来的验证集列表')
    #这两个if表示，如果提前存在就删除'''
    ary = gdal.Open(file).ReadAsArray()
    cirrus = []
    smoke = []
    others = []
    for i in range(ary.shape[1]):
        for j in range(ary.shape[2]):
            if ary[-1,i,j]>0:
                smoke.append(ary[:,i,j])
            elif 0.2 < ary[0, i, j] < 0.24:
                cirrus.append(ary[:,i,j])
            else:
                others.append(ary[:,i,j])
    random.shuffle(cirrus)
    random.shuffle(smoke)
    random.shuffle(others)
    # 每幅图选300个卷云样本,选500个烟样本,选200个其它样本 20210129
    # 每幅图选200个卷云样本，选400个烟样本，选400个其它样本 20210130
    # 每幅图直选
    cirrus_selected = cirrus[:200]
    smoke_selected = smoke[:300]
    others_selected = others[:3000]
    total = cirrus+smoke+others
    random.shuffle(total)
    td = smoke_selected[:200] + others_selected[:1800] #每幅图xx个像元用作训练
    #td = total[:2000]
    vd = smoke_selected[-100:] + others_selected[-900:] #每幅图xx个像元用作验证
    #vd = total[-1000:]
    np.save(td_pth,td)
    np.save(vd_pth,vd)

i=1
for date in ['0817','0821','0823','0825']:
    for p in glob.glob(r'E:\SmokeDetection\source\new_new_data\{}\*.tif'.format(date)):
        create_new_index(p,r'E:\SmokeDetection\source\training_pixels\{}\{}_td.npy'.format(date,p[-8:-4]) ,\
                r'E:\SmokeDetection\source\training_pixels\{}\{}_vd.npy'.format(date, p[-8:-4]))
        print('搞定{}个'.format(i))
        i=i+1