clc
clear
% nc�ļ���·��
ncFileBasePath='I:\�̼������д��\other dataset\1ԭʼ����\';
% ���tif��·��
tiffOutFileBasePath='I:\�̼������д��\other dataset\1ԭʼ����\';
% ��ȡ���ļ���������nc��
imageList=dir(strcat(ncFileBasePath,'*.nc'));

for fileindex=1:length(imageList)
   % �����ļ���
   filename=imageList(fileindex).name;
   % ������׺���ļ���
   filenameWithoutSufix=filename(1:find(filename=='.')-1);
   % nc�ļ�������·��
   fileFullPath=strcat(ncFileBasePath,filename);
   
   ncinf = ncinfo(fileFullPath);
  
   ETaSets=ncread(fileFullPath,);
   
   SizeInfo=size(ETaSets);
        for subsetIndex=1:SizeInfo(3)
        
           disp(subsetIndex);
           MonthData=ETaSets(:,:,subsetIndex);
           MonthData(isnan(MonthData))=-8888;
           tifOutputFullPath=strcat(tiffOutFileBasePath,filenameWithoutSufix,num2str(subsetIndex,'%02d'),'.tif');
           MonthData = rot90(MonthData);
           Refference=georasterref('RasterSize',size(MonthData),'Latlim',[-89.75 89.75],'Lonlim',[-179.75 179.75]); 
           Refference.ColumnsStartFrom = 'north';
           geotiffwrite(tifOutputFullPath,MonthData,Refference);
           
        end
   
end
disp('�ɹ���');