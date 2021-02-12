#找出这组数据不同波段的最大值和最小值
import glob
import gdal
import numpy as np

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