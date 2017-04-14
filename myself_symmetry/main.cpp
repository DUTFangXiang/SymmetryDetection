/*********************************************************************
FX
Time:           From 2015.4.27
Reference: 2010 ICPR Using Symmetry to Select Fixation Points for Segmentation
platform :   Visual Studio 2010 x86  opencv 2.4.8
*********************************************************************/
/////////////////////////////////////////////////////////////////////
//openvc���ã�
//   ������win7  VC2010 + opencv2.4.8
//  1��Project -> Properties -> VC++director����include library��
//��� ��%opencv%\build\include��
//  2��Project -> Properties -> VC++director����library director��
//����С�%opencv%\build\x86\vc10\lib��
//  3��Project ->Properties->Linker-> input ->additional dependencies
//��ӣ�opencv_calib3d240d.lib ��opencv_contrib240d.lib
//opencv_core240d.lib��opencv_features2d240d.lib
//opencv_flann240d.lib��opencv_gpu240d.lib��opencv_highgui240d.lib
//opencv_imgproc240d.lib��opencv_legacy240d.lib��opencv_ml240d.lib
//opencv_objdetect240d.lib��opencv_ts240d.lib��opencv_video240d.lib
// ��������һ�����⣬������opencv2.4.8�����Դ˴�Ҫ�޸ĳ�240 ��>248
//   4�����ڿ���ֱ�Ӽ��������õĹ������Ա���F:\english_surface\opencv\
//opencv248.props 
//  
//   5�����Թر�warning��Alt+F7,��Project Setting�Ի��򣬵�C++ѡ���
//Warning level ,ѡtake off   
////////////////////////////////////////////////////////////////////
//�޸���־��
//  1��<unistd.h>��linux�ӿ�ͷ�ļ���windows�������
//�Խ�һ����ͷ�ļ��� 
//  2��<sys/time.h>��windows��vc����sysֻ��utime.h;
//  3������#define M_PI 3.14159265358979323846��
// vc2010�汾��û�иö��壬��ʱ����additional.h
//  4��sqrt()Ҫ������Ϊfloat����
//  5��û��DIRĿ¼����ͷ
//  6��rint����������vc�У���Ҫ�Լ�д,������additional.cpp����
//  7��project->xx Properties -> Manifest->Input and Output
//->Embed Manifest�޸ĳ� NO
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
  const char imgFilepath[256] = "..\\imageMSRA\\0_3_3514";    //����û�С�.jpg����β
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
