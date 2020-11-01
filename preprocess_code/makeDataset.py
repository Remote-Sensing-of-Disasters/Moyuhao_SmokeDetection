# 把风向数据、葵花数据、葵花标签这三个tif文件合成一个tif文件，左上角坐标是东经90°，北纬25°
import gdal
import glob
import os
import time as t
def mk_data(ary, fn_wind_u, fn_wind_v, label, name):
    def normalize(ary,wind_u,wind_v):
        #所有通道都进行一次拉伸，拉伸到[0,1]
        for i in range(ary.shape[0]-1):
            chl = ary[i,:,:]
            ary[i,:,:] = (chl-chl.min())/(chl.max()-chl.min()+1e-7)
        wind_u = (wind_u-wind_u.min())/(wind_u.max()-wind_u.min()+1e-7) #风向数据归一化
        wind_v = (wind_v-wind_v.min())/(wind_v.max()-wind_v.min()+1e-7)

        return ary, wind_u, wind_v

    def combine(ary, wind_u, wind_v , label, name):
        ary, wind_u, wind_v = normalize(ary, wind_u, wind_v )
        driver=gdal.GetDriverByName('GTiff')
        out = driver.Create('{}.tif'.format(name),2500,1750,19,gdal.GDT_Float32)
        for i in range(16):
            out.GetRasterBand(i+1).WriteArray(ary[i,1:,1:])
        out.GetRasterBand(i+2).WriteArray(wind_u)
        out.GetRasterBand(i+3).WriteArray(wind_v)
        out.GetRasterBand(i+4).WriteArray(label[1:,1:])
        out.SetGeoTransform((90,0.02,0.0,25,0.0,-0.02))
        out.SetProjection(gdal.Open(r'E:\SmokeDetection\source\20150825_for_test\test_geosamples\0000.tif').GetProjection())

    combine(ary, fn_wind_u, fn_wind_v, label, name)

    print(t.asctime())

os.chdir(r'I:\normalized_data_2020_10_25')
fns = glob.glob(r'E:\SmokeDetection\source\new_data\0823\0250.tif')#葵花数据文件名
for fn in fns:
    ary = gdal.Open(fn).ReadAsArray()
    fn_l = fn.replace(r'E:\SmokeDetection\source\new_data',r'I:\new_label')#标签文件名
    label = gdal.Open(fn_l).ReadAsArray()
    time = fn[-8:-4]
    date = fn.split('\\')[-2]
    if not time[0]=='0':
        num = eval(time[0:3])
    elif time[1] == '0':
        num = eval(time[3])
    elif not time[1]=='0':
        num = eval(time[1:3])
    if 0 <= num <= 13: sig = 0
    if 14 <= num <= 43: sig = 3
    if 44 <= num <= 73: sig = 6
    if 74 <= num <= 103: sig = 9

    #print(sig)
    #print(time)
    wu = gdal.Open(r'I:\wind08\{}_0{}_u.tif'.format(date, sig)).ReadAsArray()
    wv = gdal.Open(r'I:\wind08\{}_0{}_v.tif'.format(date, sig)).ReadAsArray()
    name = r'I:\normalized_data_2020_10_25\{}\{}'.format(date,time)
    mk_data(ary,wu,wv,label,name)
    print('跑完一个{}_{}了'.format(date,time))