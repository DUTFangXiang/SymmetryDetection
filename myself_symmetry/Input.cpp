/*
 * FAST AND AUTOMATIC DETECTION AND SEGMENTATION OF UNKNOWN OBJECTS (ADAS)
 * 
 * Author:   Gert Kootstra, Niklas Bergstrom
 * Email:    gertkootstra@gmail.com
 * Website:  http://www.csc.kth.se/~kootstra/
 * Date:     22 Jul 2010    
 *
 * Copyright (C) 2010, Gert Kootstra
 *
 * See README.txt and the html directory for documentation 
 * and terms.
 */

#include "Input.h"

Input::Input(char *imgFile,int wScaled, int hScaled) {
  widthScaled = wScaled;
  heightScaled = hScaled;
  int dispMedianSize = 3;

  // Image
  /////////获得图片的数据，3维   imgOriginal
  imgOriginal = cvLoadImage(imgFile, 1);
  widthOriginal = imgOriginal->width;
  heightOriginal = imgOriginal->height;
  //创建头并分配数据，参数意义分别是：
  //图像宽、高，图像元素的位深度；IPL_DEPTH_8U表示无符号8位整型；每个元素（像素）通道数.
  imgScaled = cvCreateImage(cvSize(widthScaled, heightScaled), IPL_DEPTH_8U, 3);
  //缩放源图像到目标图像； 
  cvResize(imgOriginal, imgScaled, CV_INTER_NN);
  //将图片转化到一个char数组
  imgDataScaled = copyImageDataFloat(imgScaled);  
  /*
  // Disparity map
  IplImage *tmpDisp = cvLoadImage(dispFile, 0);
  disp = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 1);
  cvResize(tmpDisp, disp, CV_INTER_NN);
  cvReleaseImage(&tmpDisp);
  int distMedianSize = 3;
  dispData = copyDispDataFloat(disp, distMedianSize);
  */
  // Resize proportion
  resizePropX = (float)widthScaled / widthOriginal;
  resizePropY = (float)heightScaled / heightOriginal;
 
  yCrCbImgScaled = NULL;
  yCrCbImgDataScaled = NULL;
}

Input::~Input() {
  cvReleaseImage(&imgOriginal);
  cvReleaseImageHeader(&dispOriginal);
  cvReleaseImage(&imgScaled);
  cvReleaseImage(&dispScaled);
  delete[] imgDataScaled;
  delete[] dispDataScaled;
  if(yCrCbImgScaled)
    cvReleaseImage(&yCrCbImgScaled);
  //if(yCrCbImgData) // is still element of yCrCbImg, so deleted with the previous statement
  //delete[] yCrCbImgData;
}

float *Input::readDisparityData(char *dispFile, int width, int height) {
  float *dispData = new float[width*height];
  FILE *dispFD = fopen(dispFile, "r");
  fread(dispData, sizeof(float), width*height, dispFD);
  fclose(dispFD);
  return(dispData);
}

void Input::calcLab() {
  // Lab image data
  IplImage *_imgScaled = cvCreateImageHeader(cvSize(widthScaled, heightScaled), IPL_DEPTH_32F, 3);
  cvSetData(_imgScaled, imgDataScaled, sizeof(float)*widthScaled*3);
  yCrCbImgScaled = cvCreateImage(cvSize(widthScaled, heightScaled), IPL_DEPTH_32F, 3);
  cvCvtColor(_imgScaled, yCrCbImgScaled, CV_BGR2Lab);
  yCrCbImgDataScaled = (float *)yCrCbImgScaled->imageData;
  //将图片数据转化为Lab格式数据
  /*! 
   * The practical limits of a and b in Lab space are experimentally 
   * determined to be:
   *  - a: [-86.181, 98.235]
   *  - b: [-107.861, 94.475]
   */
  int labI;
  float aOffset = 87.0f;
  float aRange = 87.0f + 99.0f;
  float bOffset = 108.0f;
  float bRange = 108.0f + 95.0f;
  for(int y=0; y<heightScaled; y++) {
    for(int x=0; x<widthScaled; x++) {
      labI = 3*(y*widthScaled+x);
      
      yCrCbImgDataScaled[labI] /= 100;
      yCrCbImgDataScaled[labI+1] += aOffset;
      yCrCbImgDataScaled[labI+1] /= aRange;
      yCrCbImgDataScaled[labI+2] += bOffset;
      yCrCbImgDataScaled[labI+2] /= bRange;
    }
  }
  cvReleaseImageHeader(&_imgScaled);
}

float *Input::copyImageDataFloat(IplImage *img) {
  // Copies the image data into a char array, so that all pixels are nicely aligned
  int width = img->width;
  int height = img->height;
  int widthStep = img->widthStep;
  int nChannels = img->nChannels;
  uchar *imgData = (uchar *)img->imageData;
  float *copyData = new float[width*height*nChannels];

  int iI, cI;
  for(int y=0; y<height; y++) {
    for(int x=0; x<width; x++) {
      iI = y*widthStep + x*nChannels;
      cI = y*width*nChannels + x*nChannels;
      for(int c=0; c<nChannels;c++) {
        copyData[cI+c] = (float)imgData[iI+c]/255;
      }
    }
  }
  return copyData;
}

float *Input::copyDispDataFloat(IplImage *_dImg, int distMedianSize) {
  // Smooth the disparity image to get rid of some noise
  int width = _dImg->width;
  int height = _dImg->height;
  IplImage *dImg = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 1);
  cvSmooth(_dImg, dImg, CV_MEDIAN, distMedianSize);

  // Get all uchars and scale them between 0 and 1
  // NB. The maximum valid disparity value in our case in 63
  int widthStep = dImg->widthStep;
  uchar *dispData = (uchar *)dImg->imageData;
  float *copyData = new float[width*height];

  int iI, cI;
  for(int y=0; y<height; y++) {
    for(int x=0; x<width; x++) {
      iI = y*widthStep + x;
      cI = y*width + x;
      if(dispData[iI]==(uchar)255)
        copyData[cI] = -1;
      else
        copyData[cI] = (float)dispData[iI]/63;
    }
  }
  cvReleaseImage(&dImg);

  return copyData;
}

float *Input::copyDispDataFloat2(IplImage *_dImg, int distMedianSize) {
  // Smooth the disparity image to get rid of some noise
  int width = _dImg->width;
  int height = _dImg->height;
  IplImage *dImg = cvCreateImage(cvSize(width, height), IPL_DEPTH_32F, 1);
  cvSmooth(_dImg, dImg, CV_MEDIAN, distMedianSize);

  // Get all uchars and scale them between 0 and 1
  // NB. The maximum valid disparity value in our case in 63
  float *dispData = (float *)dImg->imageData;
  float *copyData = new float[width*height];

  int iI, cI;
  for(int y=0; y<height; y++) {
    for(int x=0; x<width; x++) {
      iI = y*width + x;
      cI = y*width + x;
      if(dispData[iI]<0.0f)
	copyData[cI] = -1;
      else
	copyData[cI] = dispData[iI]/64.0f;
    }
  }
  cvReleaseImage(&dImg);

  return copyData;
}

IplImage *Input::getDispImage() {
  IplImage *disp = cvCreateImage(cvSize(widthScaled, heightScaled), IPL_DEPTH_8U, 3);
  uchar * dispData = (uchar *)disp->imageData;
  Color color;
  float r,g,b;
  int i, jetI;
  for(int y=0; y<heightScaled; y++) {
    for(int x=0; x<widthScaled; x++) {
      i = y*widthScaled+x;
      if(dispDataScaled[i] >= 0.0f) {
	jetI = (int)(dispDataScaled[i] * color.jetSize);
	//printf("%1.2f: %d\n", disp, jetI);
	r = color.jetU8[3*jetI];
	g = color.jetU8[3*jetI+1];
	b = color.jetU8[3*jetI+2];
      }
      else {
	r = (uchar)0;
	g = (uchar)0;
	b = (uchar)0;
      }
      dispData[3*i] = b;
      dispData[3*i+1] = g;
      dispData[3*i+2] = r;
    }
  }
  return disp;
}

