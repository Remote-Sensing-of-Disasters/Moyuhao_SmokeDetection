import matplotlib.pyplot as plt
import numpy as np
import os
def out1line(root, file, label, title):
    os.chdir(root)
    with open(file) as f:
        aaa = f.readlines()
    para_a = []
    for a in aaa:
        para_a.append(eval(a[:-1]))
    x = np.linspace(1, len(para_a), len(para_a), dtype=np.int)
    plt.plot(x, para_a, color='b', label=label)
    plt.legend()
    plt.title(title)
    plt.xlabel('Epoch')
    plt.show()

roots=[]
roots.append(r'E:\SmokeDetection\source\MLP\result\72_10') # 0
roots.append(r'E:\SmokeDetection\source\MLP_Results\256_3')  # 1
roots.append(r'E:\SmokeDetection\source\MLP_Results\256_3_NoBN') # 2
root = roots[1]
cat = [
    ['loss_td.txt', 'Training Data Loss', 'Training Loss'],  #训练数据的损失 0
    ['accuracy_td.txt', 'Training Data OA', 'Training OA'],  #训练数据的精度 1
    ['precision_td.txt', 'Training Data Precision', 'Training Precision'],  #训练数据的查准率 2
    ['recall_td.txt', 'Training Data Recall', 'Training Recall'],  #训练数据的查全率 3
    ['iou_td.txt', 'Training Data IoU', 'Training IoU'],  #训练数据的IOU 4
    ['loss_vd.txt', 'Validation Data Loss', 'Validation Loss'], #验证数据的损失 5
    ['accuracy_vd.txt', 'Validation Data OA', 'Validation OA'],  #验证数据的精度 6
    ['precision_vd.txt', 'Validation Data Precision', 'Validation Precision'],  #验证数据的查准率 7
    ['recall_vd.txt', 'Validation Data Recall', 'Validation Recall'],  #验证数据的查全率 8
    ['iou_vd.txt', 'Validation Data IoU', 'Validation IoU']  #验证数据的IOU 9
       ]
file , label , title = cat[0]
if 0:  out1line(root, file, label, title)

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