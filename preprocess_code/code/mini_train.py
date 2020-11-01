from batch_train import train
import glob
fns = glob.glob(r'E:\SmokeDetection\source\MLP\result\*')#不同框架的MLP的存储路径：命名形式:每层节点数_层数
for fn in fns[:1]:
    num_layer = eval(fn.split('\\')[-1].split('_')[-1])
    hid_vertex = eval(fn.split('\\')[-1].split('_')[-2])
    train(r'E:\SmokeDetection\source\new_samples_64',r'E:\SmokeDetection\source\new_samples_64',fn,18,2,hid_vertex,num_layer,100,1,1e-4)
print('跑完了')
#7.21，17：19 更改tdloss 注释val_data代码，只抽取8个样本作为训练，Adam

#先看少数据能否收敛，之后再考虑用全数据做一次
#7.22 miniresult跑起来，迭代500次

#7.30 改掉random.shuffle
#8.6改成了batch size为文件动态的，Adam改成SGD加momentum0.8
#8.6又改回了Adam，参数全默认
#8.9只运行有烟像元直到训练至100   [648, 36, 72, 2, 1296, 36]
