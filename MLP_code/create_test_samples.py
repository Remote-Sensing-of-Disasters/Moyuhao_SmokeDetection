import os
import glob
import random
import gdal
import numpy as np
import time

def create_by_random(file, td_pth, num):
    def sieve_tvdata(data, smp_pth, net_file):
        # 把每个时刻的影像中的训练和验证样本像元删除，剩下的就是测试样本
        print(time.asctime())
        smpType = net_file.split('\\')[-3]
        rdm = net_file.split('\\')[-2]
        date = smp_pth.split('\\')[-2]
        datetime = smp_pth.split('\\')[-1][:4]
        tddata_pth = r'E:\SmokeDetection\source\MLP_90_dataset\{}\{}\{}\{}_td.npy'.format(smpType, rdm, date, datetime)
        vddata_pth = r'E:\SmokeDetection\source\MLP_90_dataset\{}\{}\{}\{}_vd.npy'.format(smpType, rdm, date, datetime)
        tddata = np.load(tddata_pth)
        vddata = np.load(vddata_pth)
        tvdata = np.r_[tddata, vddata]
    ary = gdal.Open(file).ReadAsArray()
    total=[]
    for i in range(ary.shape[1]):
        for j in range(ary.shape[2]):

            total.append(ary[:, i, j])

    random.shuffle(total)
    td = total[:num]  # 每幅图xx个像元用作测试

    np.save(td_pth, td)



fns = glob.glob(r'E:\SmokeDetection\source\test_fourdays_data_samples\*')
for fn in fns:
    for date in ['0817', '0821', '0823', '0825']:  #['0817', '0821', '0823', '0825']
        for p in glob.glob(r'E:\SmokeDetection\source\new_new_data\{}\*.tif'.format(date)):
            create_by_random(p, fn+r'\{}\{}_test.npy'.format(date, p[-8:-4]), 1000)
    print('搞定{}文件夹'.format(fn))




'''if not os.path.exists(fn+r'\{}'.format(date)):
    os.mkdir(fn+r'\{}'.format(date))
    print(fn+r'\{}'.format(date))'''
