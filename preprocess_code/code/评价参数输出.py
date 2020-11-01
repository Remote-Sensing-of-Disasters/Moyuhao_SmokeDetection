import matplotlib.pyplot as plt
import numpy as np
import os
roots=[]
roots.append(r'E:\SmokeDetection\source\MLP\result\72_10') # 0
roots.append(r'E:\SmokeDetection\source\MLP\result\36_3')  # 1
roots.append(r'E:\SmokeDetection\source\MLP\result\128_20') # 2
root = roots[2]
#输出一个OA曲线
if 1:
    with open(root+r'\accuracy_td.txt') as f:
        aaa=f.readlines()
    para_a=[]
    for a in aaa:
        para_a.append(eval(a[:-1]))
    x=np.linspace(1,len(para_a),len(para_a),dtype=np.int)
    plt.plot(x,para_a,color='b',label='Training Data OA')
    plt.legend()
    plt.title('OA')
    plt.xlabel('Epoch')
    plt.show()

#输出一个Loss 曲线
if 1:
    with open(root+r'\loss_td.txt') as f:
        aaa=f.readlines()
    para_a=[]
    for a in aaa:
        para_a.append(eval(a[:-1]))
    x=np.linspace(1,len(para_a),len(para_a),dtype=np.int)
    plt.plot(x,para_a,color='b',label='Training Data Loss')
    plt.legend()
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.show()



#输出两个曲线
if 1:
    with open(root+r'\precision_td.txt') as f:
        aaa=f.readlines()
    with open(root+r'\recall_td.txt') as f:
        bbb=f.readlines()

    para_a=[]
    for a in aaa:
        para_a.append(eval(a[:-1]))

    para_b=[]
    for b in bbb:
        para_b.append(eval(b[:-1]))

    x=np.linspace(1,len(para_b),len(para_b),dtype=np.int)
    plt.plot(x,para_a,color='r',label='Training Data precision')
    plt.plot(x,para_b,color='b',label='Training Data recall')
    plt.legend()
    plt.title('P & R')
    plt.xlabel('Epoch')
    #plt.ylabel('')
    plt.show()



#输出3个曲线
'''with open(r'E:\SmokeDetection\source\LSTM\train_result\recall_vd.txt') as f:
    aaa=f.readlines()
with open(r'E:\SmokeDetection\source\LSTM\train_result\precision_vd.txt') as f:
    bbb=f.readlines()

with open(r'E:\SmokeDetection\source\LSTM\train_result\recall_td.txt') as f:
    ccc = f.readlines()
with open(r'E:\SmokeDetection\source\LSTM\train_result\precision_td.txt') as f:
    ddd = f.readlines()

para_a=[]
for a in aaa:
    para_a.append(eval(a[:-2]))
para_a=np.array(para_a)
para_b=[]
for b in bbb:
    para_b.append(eval(b[:-2]))
para_b=np.array(para_b)
para_c=[]
for c in ccc:
    para_c.append(eval(c[:-2]))
para_c=np.array(para_c)
para_d=[]
for d in ddd:
    para_d.append(eval(d[:-2]))
para_d=np.array(para_d)

f1_vd=2*para_a*para_b/(para_a+para_b)
f1_td=2*para_c*para_b/(para_c+para_d)
x=np.linspace(1,len(para_b),len(para_b),dtype=np.int)
plt.plot(x,f1_vd,color='r',label='Validation Data F1')
plt.plot(x,f1_td,color='b',label='Training Data F1')
plt.xlabel('Epoch')
plt.ylabel('F1')
plt.legend()
plt.show()'''