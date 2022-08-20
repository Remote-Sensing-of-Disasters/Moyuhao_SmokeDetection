#找出这组数据不同波段的最大值和最小值
import glob
import gdal
import numpy as np

# fns_data = glob.glob(r'E:\SmokeDetection\source\MLP_cirrus_test\50000_smoke2500\Morning_20220703\*\*.npy')
# for fn in fns_data:
#     data = np.load(fn)
#     time = np.load(r'E:\SmokeDetection\source\MLP_cirrus_test\50000_smoke2500\Morning_20220622\{}\{}'.format(fn.split('\\')[-2],fn.split('\\')[-1]))
#     data[:,16]=time[:,16]
#     np.save(fn,data)

fns = glob.glob(r'E:\SmokeDetection\source\new_new_data\*\*.tif')
Min = np.ones([18])*1000
Max = np.ones([18])*(-1000)
for fn in fns:
    ary = gdal.Open(fn).ReadAsArray()
    for i in range(18):
        if ary[i,:,:].max()>Max[i]:
            Max[i]=ary[i, :, :].max()
        if ary[i,:,:].min()<Min[i]:
            Min[i]=ary[i,:,:].min()
print(Min)
print(Max)

# for b in range(18):
#     print('第{}个波段的均值：{}，标准差：{}'.format(b+1,temp[b,:,:].mean(),temp[b,:,:].std()))
#     a.append([temp[b,:,:].mean(),temp[b,:,:].std()])