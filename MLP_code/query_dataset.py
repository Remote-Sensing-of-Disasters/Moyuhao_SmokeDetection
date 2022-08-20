import glob
import numpy as np


vd_fns = glob.glob(r'E:\SmokeDetection\source\MLP_cirrus_test\4000_cirrus\0\*\*vd.npy')
for fn in vd_fns:
    data = np.load(fn)
    if not data.shape[0]==2000:
        print(fn)
    #print(data.shape[0])
