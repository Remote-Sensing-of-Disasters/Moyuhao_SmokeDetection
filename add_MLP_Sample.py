#将700个地面非烟像元纳入训练样本中
#0817 0010 左上：[393,216] 右下：[453,243]
import gdal
import numpy as np
import os
os.chdir(r'E:\SmokeDetection\source\training_pixels\0817')
fn = r'E:\SmokeDetection\source\new_new_data\0817\0010.tif'
ary = gdal.Open(fn).ReadAsArray()
out = []
for i in range(393,453):
    for j in range(216,243):
        out.append(ary[:,i,j])
np.save('1000_td.npy',out)
