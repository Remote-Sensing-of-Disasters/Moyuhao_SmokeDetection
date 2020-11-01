#将对应日期的大图按照相同的行列号采样成64*64大小的小图
import gdal
import os
import numpy as np
import glob
import random
import time as t

def add_wind(name):
    '''name是葵花影像成像日期和UTC时刻，如0817_0000,该函数返回对应的风向图两张'''
    date = name.split('_')[-2]
    time = name.split('_')[-1]
    if time[1]=='0':
        num = eval(time[3])
    else:
        num = eval(time[1:3])
    if 0<=num<=13:sig=0
    if 14<=num<=43:sig=3
    if 44<=num<=73:sig=6
    if 74 <= num <= 103: sig =9
    fnu = r'I:\烟检测论文写作\other dataset\1原始数据\1风向数据\{}\u{}.tif'.format(date, sig)
    fnv = r'I:\烟检测论文写作\other dataset\1原始数据\1风向数据\{}\v{}.tif'.format(date, sig)

    u,v = gdal.Open(fnu).ReadAsArray(), gdal.Open(fnv).ReadAsArray()
    return u, v
'''
def normalize(ary,wind_u,wind_v):
    #光谱和风向数据归一化
    bt = ary[6:16,:,:]
    ary[6:16,:,:] = ((bt-180)/(401-180))+1e-7
    wind_u = ((wind_u+10)/20)+1e-7
    wind_v = ((wind_v+10)/20)+1e-7
    return ary, wind_u, wind_v
'''
def mk_pic(ary, save_pth, name, row, col, size):
    '''ary是单幅图像，save_pth是图片保存文件夹路径， name是图片成像日期以及时刻，row和col是采样图片左上角行列号'''
    u,v = add_wind(name)
    #ary,u,v=normalize(ary,u,v)
    driver = gdal.GetDriverByName('GTiff')
    out = driver.Create(save_pth+r'\{}_{}_{}_nosmoke.tif'.format(name, row, col), size, size, ary.shape[0]+2, gdal.GDT_Float32)
    for i in range(ary.shape[0]-1): #把光谱影像加进来
        out.GetRasterBand(i+1).WriteArray(ary[i,row:row+size,col:col+size])
    out.GetRasterBand(17).WriteArray(u[row:row+size,col:col+size])
    out.GetRasterBand(18).WriteArray(v[row:row+size,col:col+size])
    out.GetRasterBand(19).WriteArray(ary[-1,row:row+size,col:col+size])
    sample_img = gdal.Open(r'E:\SmokeDetection\source\20150825_for_test\test_geosamples\0010.tif')
    geot = sample_img.GetGeoTransform()
    geot = list(geot)
    geot[0] = geot[0] + geot[1] * col
    geot[3] = geot[3] + geot[5] * row
    geot = tuple(geot)
    out.SetGeoTransform(geot)
    out.SetProjection(sample_img.GetProjection())
    del out

def smp_day(fn, save_pth, pix_dirs):
    '''fn 是文件所在的文件夹，save_pth是图片保存文件夹，pix_dirs是采样图像左上像素行列号集合'''
    smp = glob.glob(fn+r'\*.tif')
    for s in smp:
        ary = gdal.Open(s).ReadAsArray()
        name = s.split('\\')[-2] + '_' + s.split('\\')[-1][:4]
        for [row,col] in pix_dirs:
            mk_pic(ary, save_pth, name, row, col, 64)

def find_idx(date, smoke):
    '''输入日期，输出小图的左上角坐标，smoke=1则取有烟的样本，smoke=0取无烟的样本'''

    string='smoke'
    fns = glob.glob(r'I:\海陆比\plan-1\plan04\index\{}\{}\*.txt'.format(date, string))
    idx = []
    for fn in fns:
        with open(fn, 'r') as f:
            index = f.readlines()
        index = eval(index[0])
        if len(index) == 0:
            continue
        for i in index:
            if i not in idx: idx.append(i)
    if smoke == 0:
        ns_index = []
        for i in range(0,2501,64):
            for j in range(0,1751,64):
                if [i,j] not in idx:
                    ns_index.append([i,j])
        random.shuffle(ns_index)
        idx = ns_index[:20]
    return idx

def check_pic_smoke(ary, row, col, size, has_smoke):
    '''ary是原图，整张大图，row,col是想要调查的小图的左上角行列号，size是小图的边长，has_smoke是问你是要检查它有烟呢还是无烟呢
    如果要求它检查有烟，那has_smoke参数设置为1，如果要求检查小图内无烟，就设置为0。
    输出True/Flase表示是否符合has_smoke里提出的要求。
    '''
    label = ary[-1,row:row+size,col:col+size]#调用出它的标签页
    if has_smoke == 0:
        if label.sum()==0:
            return True
        else:
            return False
    if has_smoke == 1:
        if label.sum()>0:
            return True
        else:
            return False

def clip(ary, pth, name, row, col, size=64):
    '''将new data裁剪成64*64的小图，ary是小图，pth是保存路径，name是文件名，row,col是左上点行列号，size是图的边长64'''
    driver = gdal.GetDriverByName('GTiff')
    out = driver.Create(pth + name, size, size, ary.shape[0],
                        gdal.GDT_Float32)
    for i in range(ary.shape[0]):
        out.GetRasterBand(i+1).WriteArray(ary[i,row:row+size,col:col+size])
    sample_img = gdal.Open(r'E:\SmokeDetection\source\20150825_for_test\test_geosamples\0010.tif')
    geot = sample_img.GetGeoTransform()
    geot = list(geot)
    geot[0] = geot[0] + geot[1] * col
    geot[3] = geot[3] + geot[5] * row
    geot = tuple(geot)
    out.SetGeoTransform(geot)
    out.SetProjection(sample_img.GetProjection())
    del out


if __name__ == '__main__':
    #将new data中的小图全都裁剪下来
    fns = glob.glob(r'I:\normalized_data_2020_10_25\0823\0250.tif')
    save_pth = r'I:\samples_2020_10_25'
    for fn in fns:
        date = fn.split('\\')[-2]
        time = fn.split('\\')[-1][:4]
        pth = save_pth+'\\'+date
        ary = gdal.Open(fn).ReadAsArray()
        Is = list(range(0,1750,64))
        Is.append(1750-64)
        Js = list(range(0,2500,64))
        Js.append(2500-64)
        for i in Is:
            for j in Js:
                if ary[-1,i:i+64,j:j+64].sum()>0:
                    name = r'{}_{}_{}_smoke.tif'.format(time, i, j)
                else:
                    name = r'{}_{}_{}_nosmoke.tif'.format(time, i, j)
                clip(ary,pth,name,i,j)
        print('筛选完{}_{}图了'.format(date,time))
        print(t.asctime())