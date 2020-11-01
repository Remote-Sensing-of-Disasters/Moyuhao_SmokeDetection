clc
clear
% nc文件根路径
ncFileBasePath='I:\烟检测论文写作\other dataset\1原始数据\';
% 输出tif根路径
tiffOutFileBasePath='I:\烟检测论文写作\other dataset\1原始数据\';
% 获取该文件夹下所有nc文
imageList=dir(strcat(ncFileBasePath,'*.nc'));

for fileindex=1:length(imageList)
   % 完整文件名
   filename=imageList(fileindex).name;
   % 不带后缀的文件名
   filenameWithoutSufix=filename(1:find(filename=='.')-1);
   % nc文件的完整路径
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
disp('成功！');