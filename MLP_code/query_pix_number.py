#查询图中对应标签的像元数目
import numpy as np
import gdal
import glob
from tqdm import tqdm
# def query(ary,label=255):
#     #ary是图的np格式，label是想查询的标签名
#     temp = np.where(ary[:,:]==label,1,0) #默认label在最后一个标签页
#     sum = temp.sum() #返回这张图中带有标签的像元的数量
#     return sum
#
#
# fns = glob.glob(r'E:\SmokeDetection\source\semi-supervised learning\cliped_new_new_data\0830\*.tif')
# sum_1 = 0
# sum_2 = 0
# for fn in fns:
#     ary = gdal.Open(fn).ReadAsArray()
#     sum_2 +=query(ary,0)
#     sum_1 += query(ary,1)
#     print(fn)
# print(sum_1)
# print(sum_1/sum_2)
# print(sum_1/(sum_1+sum_2))
# print(len(fns))
#

#所有samples_64数据，1185张图中，有烟像元381979个，无烟像元4471781个


#
f=glob.glob(r'E:\SmokeDetection\source\new_new_data\*\00*0.tif')
Dataset_train = glob.glob(r'E:\SmokeDetection\source\MLP_cirrus_test\50000_smoke2500\Morning_20220530\*\*_td.npy')
fns = [r'E:\SmokeDetection\source\new_new_data\{}\{}.tif'.format(data.split('\\')[-2],data.split('\\')[-1][:4]) for data in Dataset_train]

fnsMorning = glob.glob(r'E:\SmokeDetection\source\semi-supervised learning\new_new_data\*\00*0.tif')
fns0100 = glob.glob(r'E:\SmokeDetection\source\semi-supervised learning\new_new_data\*\0100.tif')
fns = fnsMorning+fns0100
# fns = glob.glob(r'E:\SmokeDetection\source\new_new_data\*\00*0.tif')
# for morning in f:
#     fns.remove(morning)
#print(fns)


if 0:
    temp = gdal.Open(fns[0]).ReadAsArray()[:18,:,:]#由于有的文件数据有标签波段，无法拼贴，就给他去掉
    for fn in tqdm(fns[1:]):
        ary = gdal.Open(fn).ReadAsArray()[:18,:,:]
        temp=np.c_[temp,ary]
    smoke= []

    a=[]
    for b in range(18):
        print('第{}个波段的均值：{}，标准差：{}'.format(b+1,temp[b,:,:].mean(),temp[b,:,:].std()))
        a.append([temp[b,:,:].mean(),temp[b,:,:].std()])
    print(a)
# b=0
# for bands in [6,16,18]:
#     print('mean={},std={}'.format(temp[b:bands,:,:].mean(),temp[b:bands,:,:].std()))
#     a.append([temp[b:bands, :, :].mean(), temp[b:bands, :, :].std()])
#     b=bands
#print(a)

fns = glob.glob(r'E:\SmokeDetection\source\MLP_cirrus_exp\50000_smoke2500\NoonDropoutBack\*.tif')