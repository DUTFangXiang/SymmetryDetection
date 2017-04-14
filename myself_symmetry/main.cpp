/*********************************************************************
FX
Time:           From 2015.4.27
Reference: 2010 ICPR Using Symmetry to Select Fixation Points for Segmentation
platform :   Visual Studio 2010 x86  opencv 2.4.8
*********************************************************************/
/////////////////////////////////////////////////////////////////////
//openvc配置：
//   环境：win7  VC2010 + opencv2.4.8
//  1、Project -> Properties -> VC++director：“include library”
//添加 ”%opencv%\build\include”
//  2、Project -> Properties -> VC++director：“library director”
//添加行”%opencv%\build\x86\vc10\lib”
//  3、Project ->Properties->Linker-> input ->additional dependencies
//添加：opencv_calib3d240d.lib 、opencv_contrib240d.lib
//opencv_core240d.lib、opencv_features2d240d.lib
//opencv_flann240d.lib、opencv_gpu240d.lib、opencv_highgui240d.lib
//opencv_imgproc240d.lib、opencv_legacy240d.lib、opencv_ml240d.lib
//opencv_objdetect240d.lib、opencv_ts240d.lib、opencv_video240d.lib
// 后来发现一个问题，我们是opencv2.4.8，所以此处要修改成240 —>248
//   4、现在可以直接加载我配置的工程属性表，在F:\english_surface\opencv\
//opencv248.props 
//  
//   5、调试关闭warning，Alt+F7,打开Project Setting对话框，点C++选项卡，
//Warning level ,选take off   
////////////////////////////////////////////////////////////////////
//修改日志：
//  1、<unistd.h>是linux接口头文件，windows可以添加
//自建一个该头文件； 
//  2、<sys/time.h>在windows的vc下面sys只有utime.h;
//  3、加入#define M_PI 3.14159265358979323846。
// vc2010版本中没有该定义，此时加入additional.h
//  4、sqrt()要求数据为float类型
//  5、没有DIR目录处理头
//  6、rint函数不存在vc中，需要自己写,放在了additional.cpp中了
//  7、project->xx Properties -> Manifest->Input and Output
//->Embed Manifest修改成 NO
//  
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <limits>
#include <sys/utime.h>
#include <vector>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include "unistd.h"
#include "Input.h"
#include "Symmetry.h"

using namespace std;
using namespace Saliency;
/*
 * @see See SymParams for a description of the symmetry paramters
 */
void setSymmetryParams(Symmetry *symmetry) {
  SymParams symParams;
  int symScales[] = {3,4,5}; 
  symParams.userScales = symScales;
  symParams.scales = NULL;
  symParams.nrScales = sizeof(symScales)/sizeof(int);
  symParams.r1 = 3;
  symParams.r2 = 8;
  symParams.s = 16;
  symParams.symmetryType = SYMTYPE_ISOTROPIC;
  symParams.symmetryPeriod = SYMPERIOD_2PI;
  symParams.symmetrySource = SYMSOURCE_BRIGHTNESSGRADIENT;
  symParams.calcStrongestRadius = true;
  symParams.seedR = 1;
  symParams.objThresholdAbsolute = false; // absolute or relative, max value is 1.7-2.1
  symParams.seedT = 0.5;
  symParams.roiT = 0.5;
  symmetry->setParameters(symParams);
}
//! Example program finding fixation points in an image using symmetry
/*!
 * The program reads an image, calculates the symmetry-saliency map and finds the 
 * fixation points in the map.
 * Usage: symmetrySaliency <imageFile>
 */
int main(int argc, const char* argv[]) {
  int  widthSP = 320;         
  int  heightSP = 240;       
  int  numFixations = 5;      
  int  imgWidth, imgHeight;
  char outpath[256];
  const char imgFilepath[256] = "..\\imageMSRA\\0_3_3514";    //输入没有“.jpg”结尾
  char imgFile[256];   
  IplImage *img;
  IplImage *mapresult;

  sprintf(imgFile,"%s.jpg",imgFilepath);

  img = cvLoadImage(imgFile, 1);        
  if(img==NULL) {
    printf("[ERROR] Failed to load image\n");
    exit(1);
  }
  imgWidth = img->width;                
  imgHeight = img->height;
  Input *input = new Input(imgFile,widthSP, heightSP);
  input->calcLab(); 
  // Calculate the symmetry-saliency map 
  Symmetry *symmetry = new Symmetry;
  setSymmetryParams(symmetry);
  symmetry->calcSaliencyMap(input->getImgOriginal()); 
  
  mapresult = symmetry->getSymmetryImage( );

  char lastname[15] = "_symmetry.jpg";
  sprintf(outpath ,"%s%s",imgFilepath,lastname);
  cvSaveImage (outpath, mapresult );

  return 0;
 } 
