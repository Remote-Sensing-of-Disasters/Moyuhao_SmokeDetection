#将270组评价参数分为五个页面存在一个xls表格文件里
import os
import glob
import xlwt
import xlrd

def readTXT(fns):
    #输入每个采样方式对应的10个模型在TXT文件里的评价参数，输出对应的评价参数的列表
    evlist = []
    for fn in fns:
        with open(fn,'r') as f:
            e = f.readlines()[4] #这里的序号对应了评价参数类型PRE=1,RCL=2,F1=4
            evlist.append(eval(e.split('=')[-1]))
    return evlist

def readVal(fns):
    #输入每个采样方式对应的10个模型在TXT文件里的采样测试评价参数，输出对应的评价参数列表
    evlist = []
    for fn in fns:
        with open(fn,'r') as f:
            e = f.readlines()[-1] #这里的序号取了最后一个周期的评价参数
            evlist.append(eval(e))
    return evlist

os.chdir(r'E:\SmokeDetection\source')
smpType = os.listdir(r'E:\SmokeDetection\source\MLP_NW_exp')
testType = ['NoTrain in fourdays','All in 0830']
filexlsx = xlwt.Workbook()
sheet = filexlsx.add_sheet('F1')
for i in range(len(testType)):
    sheet.write(i*10+1,0,testType[i])
for j in range(len(smpType)):
    sheet.write(0,j+1,smpType[j])

for k,t in enumerate(smpType):
    fns_4day = glob.glob(r'E:\SmokeDetection\source\MLP_NW_exp\{}\*\Evaluation report_fourdays_*_f1.txt'.format(t))
    fns_0830 = glob.glob(r'E:\SmokeDetection\source\MLP_NW_exp\{}\*\Evaluation report_0830_*_f1.txt'.format(t))
    #fns_val = glob.glob(r'E:\SmokeDetection\source\MLP_90_exp\{}\*\recall_vd.txt'.format(t))
    fourday = readTXT(fns_4day)
    test = readTXT(fns_0830)
    #val = readVal(fns_val)
    #[sheet.write((row + 1), k + 1, p) for row, p in enumerate(val)]
    [sheet.write((row + 1) ,k + 1,p) for row,p in enumerate(fourday)]
    [sheet.write((row + 1) + 10, k + 1, p) for row, p in enumerate(test)]
filexlsx.save('F1 NW.xlsx')