# SymmetryDetection
该项目是基于VS2010+Opencv2.4.8实现的图像对称性检测，可用于其它计算机视觉领域底层实现，或者对称特征提取。

图片对称性检测，越对称核心区域越高亮，根据特征梯度对称检测得到对称性分布结果。
### 检测效果图：
![](https://github.com/DUTFangXiang/SymmetryDetection/blob/master/imageMSRA/0_0_735.jpg "输入图像");
![](https://github.com/DUTFangXiang/SymmetryDetection/blob/master/imageMSRA/0_0_735_symmetry.jpg "对称检测结果");

![](https://github.com/DUTFangXiang/SymmetryDetection/blob/master/imageMSRA/0_3_3514.jpg "输入图像");
![](https://github.com/DUTFangXiang/SymmetryDetection/blob/master/imageMSRA/0_3_3514_symmetry.jpg "对称检测结果");

 ----------------------
### 调试环境：
Visual Studio 2010 x86  opencv 2.4.8

imageMSRA 文件夹中放置输入图像以及保存输出结果

-----------------------
### 实现参考论文：
2010 ICPR Using Symmetry to Select Fixation Points for Segmentation

原文是实现关注点检测，也就是对称最核心点，源程序实现在Linux下。我对源程序做了修改，移植到Windows下VS工程中，并且输出整个对称分布图。

-----------------------
### Openvc配置：
#### 环境：win7  VC2010 + opencv2.4.8
   1、Project -> Properties -> VC++director：“include library”
#### 添加 ”%opencv%\build\include”
   2、Project -> Properties -> VC++director：“library director”
#### 添加行”%opencv%\build\x86\vc10\lib”
   3、Project ->Properties->Linker-> input ->additional dependencies
添加：opencv_calib3d248d.lib 、opencv_contrib248d.lib
 opencv_core248d.lib、opencv_features2d248d.lib
 opencv_flann248d.lib、opencv_gpu240d.lib、opencv_highgui248d.lib
 opencv_imgproc248d.lib、opencv_legacy248d.lib、opencv_ml248d.lib
 opencv_objdetect248d.lib、opencv_ts248d.lib、opencv_video248d.lib
#### 关闭warning
   4、调试关闭warning，Alt+F7,打开Project Setting对话框，点C++选项卡，
 Warning level ,选take off  
 
 ----------------------
### 修改日志：
   1、<unistd.h>是linux接口头文件，windows可以添加自建一个该头文件；
   
   2、<sys/time.h>在windows的vc下面sys只有utime.h;
   
   3、加入#define M_PI 3.14159265358979323846。vc2010版本中没有该定义，此时加入additional.h
   
   4、sqrt()要求数据为float类型
   
   5、没有DIR目录处理头
   
   6、rint函数不存在vc中，需要自己写,放在了additional.cpp中了
   
   7、project->xx Properties -> Manifest->Input and Output->Embed Manifest修改成 NO
   
