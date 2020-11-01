#查询图中对应标签的像元数目
import numpy as np
import gdal
import glob
def query(ary,label=1):
    #ary是图的np格式，label是想查询的标签名
    temp = np.where(ary[-1,:,:]==label,1,0) #默认label在最后一个标签页
    sum = temp.sum() #返回这张图中带有标签的像元的数量
    return sum

'''
fn = r'E:\SmokeDetection\source\new_samples_64\0823\0000_1344_1088_smoke.tif'
ary = gdal.Open(fn).ReadAsArray()
a=ary[-1,:,:]
s=np.where(a==1)
#print(s[0])
#print(s[1])
print(a[5,36:39])

'''
'''
sum_2 =query(ary,0)
sum_1 = query(ary)
print(sum_1)
print(sum_1/(sum_1+sum_2))

'''
fns = glob.glob(r'E:\SmokeDetection\source\new_data\*\*.tif')
sum_1 =0
sum_2 = 0
for fn in fns:
    ary = gdal.Open(fn).ReadAsArray()
    sum_2 +=query(ary,0)
    sum_1 += query(ary)
#print(sum_1)
#print(sum_2)
print(sum_1/(sum_1+sum_2))
print(len(fns))


#所有samples_64数据，1185张图中，有烟像元381979个，无烟像元4471781个
