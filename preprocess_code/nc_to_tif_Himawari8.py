#把葵花8影像转换成18个通道的数据，其中前16个通道分辨对应了葵花的16个波段，后两个通道是空的，用来后面放点时间信息啥的，反正就是没有风数据了
import netCDF4 as nc
import numpy as np
import gdal
import glob
import os
# 选定一个图幅（左上角经纬度和右下角经纬度），到时候裁图就按这个做了
img = gdal.Open(r'E:\SmokeDetection\source\new_new_data\0817\0000.tif')
geotrnsfm = img.GetGeoTransform() #(107.36, 0.02, 0.0, 5.48, 0.0, -0.02)
geopjt = img.GetProjection()
shape = img.ReadAsArray().shape #(19, 501, 582)
downright = (geotrnsfm[0]+shape[2]*geotrnsfm[1],geotrnsfm[3]+shape[1]*geotrnsfm[5]) #经纬经纬，左经右纬
upleft = (geotrnsfm[0],geotrnsfm[3])
# 20211209 左上右下的经纬度分别是 (107.36, 5.48)，(119.0, -4.539999999999999)，在葵花的经纬度索引中-4.539999999999999=-4.540001,5.48=5.4799995
downright = (geotrnsfm[0]+shape[2]*geotrnsfm[1],-4.540001) #经纬经纬，左经右纬
upleft = (geotrnsfm[0],5.4799995)

temp_data = nc.Dataset(r'F:\Himawari-8\0826\NC_H08_20150826_0330_R21_FLDK.06001_06001.nc') #随便找一天葵花影像
lon = list(np.array(temp_data.variables['longitude'])) #经度
lat = list(np.array(temp_data.variables['latitude'])) #纬度
# 下面是左上角和右下角在像元中的序号，当然这是0.02°的，在不同波段还要找不同的索引
upleft_idx = (lon.index(np.float32(upleft[0])),lat.index(np.float32(upleft[1])))
downright_idx = (lon.index(np.float32(downright[0])),lat.index(np.float32(downright[1]))) # (upleft_idx,downright_idx) = (1368, 2726) (1950, 3227)
#数据中所有的波段都是同样的分辨率，0.02°


data_fns = glob.glob(r'F:\Himawari-8\*\*.nc')
for data_fn in data_fns:
    #将16个波段都读出来放进一个矩阵里头
    #data_fn = r'F:\Himawari-8\0826\NC_H08_20150826_0330_R21_FLDK.06001_06001.nc'
    data = nc.Dataset(data_fn)
    name = data_fn.split('_')[3]
    date = data_fn.split('_')[2][-4:]
    if os.path.exists(r'F:\new_new_data\{}\{}.tif'.format(date, name)):
        print('整过了{}_{}'.format(date, name))
        continue
    #16波段加载
    bands=[]
    bands.append(np.array(data.variables['albedo_01']))
    bands.append(np.array(data.variables['albedo_02']))
    bands.append(np.array(data.variables['albedo_03']))
    bands.append(np.array(data.variables['albedo_04']))
    bands.append(np.array(data.variables['albedo_05']))
    bands.append(np.array(data.variables['albedo_06']))
    bands.append(np.array(data.variables['tbb_07']))
    bands.append(np.array(data.variables['tbb_08']))
    bands.append(np.array(data.variables['tbb_09']))
    bands.append(np.array(data.variables['tbb_10']))
    bands.append(np.array(data.variables['tbb_11']))
    bands.append(np.array(data.variables['tbb_12']))
    bands.append(np.array(data.variables['tbb_13']))
    bands.append(np.array(data.variables['tbb_14']))
    bands.append(np.array(data.variables['tbb_15']))
    bands.append(np.array(data.variables['tbb_16']))

    ary = np.zeros([18,shape[1],shape[2]]).astype(np.float32) #整一个空矩阵放数据
    for i in range(16):
        ary[i,:,:] = bands[i][upleft_idx[1]:downright_idx[1],upleft_idx[0]:downright_idx[0]] #别忘了先经后纬
    # 出图
    if not os.path.exists(r'F:\new_new_data\{}'.format(date)) : os.mkdir(r'F:\new_new_data\{}'.format(date))
    os.chdir(r'F:\new_new_data\{}'.format(date))
    driver=gdal.GetDriverByName('GTiff')
    out = driver.Create('{}.tif'.format(name),shape[2],shape[1],18,gdal.GDT_Float32)
    for i in range(18):
        out.GetRasterBand(i+1).WriteArray(ary[i,:,:])
    out.SetGeoTransform(geotrnsfm)
    out.SetProjection(geopjt)
    print('成了一张图：{}_{}'.format(date,name))