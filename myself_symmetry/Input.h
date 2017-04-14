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

#ifndef INPUT_H
#define INPUT_H

#include <math.h>
#include <float.h>
#include <cstdio>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include "Color.h"

using namespace Saliency;

//! Class to read the images and disparity maps of the KOD database and transform the values to the right format and value ranges.
class Input {
 public:
  Input(char *imgFile,int widthScaled, int heightScaled);
  ~Input();
  void calcLab();
  IplImage *getImgOriginal() { return imgOriginal; };
  IplImage *getDispOriginal() { return dispOriginal; };
  IplImage *getImgScaled() { return imgScaled; };
  IplImage *getDispScaled() { return dispScaled; };
  IplImage *getYCrCbImgScaled() { return yCrCbImgScaled; };
  float *getImgDataScaled() { return imgDataScaled; };
  float *getDispDataScaled() { return dispDataScaled; };
  float *getYCrCbImgDataScaled() { return yCrCbImgDataScaled; };
  float getResizePropX() { return resizePropX; };
  float getResizePropY() { return resizePropY; };
  int getWidthOriginal() { return widthOriginal; };
  int getHeightOriginal() { return heightOriginal; };
  int getWidthScaled() { return widthScaled; };
  int getHeightScaled() { return heightScaled; };
  IplImage *getDispImage();
  
 private:
  float *readDisparityData(char *dispFile, int width, int height); 
  float *copyImageDataFloat(IplImage *img);
  float *copyDispDataFloat(IplImage *_dImg, int distMedianSize);
  float *copyDispDataFloat2(IplImage *_dImg, int distMedianSize);
  
  int widthScaled, heightScaled, widthOriginal, heightOriginal;
  IplImage *dispOriginal;
  IplImage *imgOriginal;
  IplImage *dispScaled;
  IplImage *imgScaled;
  IplImage *yCrCbImgScaled;
  float *imgDataScaled;
  float *dispDataScaled;
  float *yCrCbImgDataScaled;
  float resizePropX;
  float resizePropY;
};

#endif
