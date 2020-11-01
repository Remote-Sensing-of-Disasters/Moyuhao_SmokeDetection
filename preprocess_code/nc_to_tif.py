#nc转tif之后对图像进行双线性差值，太难了啊啊啊啊
import netCDF4 as nc
import numpy as np
import gdal
import glob
import time
#lon=90,lat=25
'''geot = (89.97999786399305, 0.019999999552965164, 0.0, 25.02000022865832, 0.0, -0.019999999552965164)
    #geo(1,2,3,4,5,6):1是左上角坐标经度，2是图像从左往右的像元中心相差多少经度，3\5是代表方位，正北就是0和0，4代表纬度,6代表向下的像元要减多少纬度（图是北纬）'''
def match_pix(data, lon, lat, x, y):
    #找到那四个值，输入单张风向数据data，给定特定data的经纬度序列,lon\lat，以及特定的经纬坐标(x,y)
    #输出该经纬坐标下的差值结果
    for l,left in enumerate(lon):
        if x-left<0.125:
            break
    right = left+0.125
    for u,up in enumerate(lat):
        if up - y <0.125:
            break
    down = up-0.125
    ul = data[u, l]
    ur = data[u, l+1]
    dl = data[u+1, l]
    dr = data[u+1, l+1]

    #总面积归一化，插值以后不会在发生像元值集体漂移出原阈
    upleft = (y-down)*(right-x)/(0.125*0.125)
    upright = (y-down)*(x-left)/(0.125*0.125)
    downleft = (up-y)*(right-x)/(0.125*0.125)
    downright = (up-y)*(x-left)/(0.125*0.125)
    output_pix = ul*upleft+ur*upright+dl*downleft+dr*downright
    return output_pix

def pix2pic(data, lon, lat):
    #输入单张风向数据图，输出单张经过双线性差值的风向数据图。Numpy 格式
    pic = np.zeros([1750,2500])
    for j,x in enumerate(np.linspace(90,90+2500*0.02,2500)):
        for i,y in enumerate(np.linspace(25,25-1750*0.02,1750)):
            pic[i,j] = match_pix(data, lon, lat, x, y)

    return pic


def bilinear(fn, date):
    #输入风向数据文件名，输入要处理的日期（不要月份）
    #输出numpy格式的文件，两个，一个u为垂直，v为水平
    data = nc.Dataset(fn)
    lon = np.array(data.variables['longitude']) #[0,-360]
    lat = np.array(data.variables['latitude']) #[90, -90]
    u10 = data.variables['u10'][date-1,:,:] #日期转换为机器语言都要减一
    ary_u10 = np.array(u10)
    v10 = data.variables['v10'][date-1,:,:]
    ary_v10 = np.array(v10)
    u = pix2pic(u10,lon, lat)
    v = pix2pic(v10,lon, lat)
    return u,v

def mk_pic(data, name):
    driver = gdal.GetDriverByName('GTiff')
    out = driver.Create(name,2500,1750,1,gdal.GDT_Float32)
    out.GetRasterBand(1).WriteArray(data)
    out.SetGeoTransform((90,0.02,0.0,25,0.0,-0.02))
    out.SetProjection(gdal.Open(r'E:\SmokeDetection\source\20150825_for_test\test_geosamples\0000.tif').GetProjection())
    del out


'''fns = glob.glob(r'I:\wind08\*.nc')[2:]
for fn in fns:
    dates = [25,30]
    for date in dates:
        u,v=bilinear(fn,date)
        mk_pic(u,r'I:\wind08\08{}_{}_u.tif'.format(date,fn[-5:-3]))
        mk_pic(v, r'I:\wind08\08{}_{}_v.tif'.format(date, fn[-5:-3]))
        print('跑完一个了')
        print(time.asctime())'''
date=30
fn=r'I:\wind08\09.nc'
u,v=bilinear(fn,date)
mk_pic(u,r'I:\wind08\08{}_{}_u.tif'.format(date,fn[-5:-3]))
mk_pic(v, r'I:\wind08\08{}_{}_v.tif'.format(date, fn[-5:-3]))
print('跑完一个了')