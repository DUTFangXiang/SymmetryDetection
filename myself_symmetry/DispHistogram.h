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

#ifndef DISPHISTOGRAM_H
#define DISPHISTOGRAM_H
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <sys/utime.h>
#include <time.h>
#include <iostream>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include "myTypes2.h"

#define uchar unsigned char

using namespace std;

//! Class containing the disparity histogram and methods to manipulate the histogram and calculate probabilities
class DispHistogram {
 public:
  DispHistogram(int nrB);
  DispHistogram(DispHistogram *dH, bool copyPDF);
  ~DispHistogram();
  void makeHistogram(float *dispData, int width, int height, int x0, int y0, int x1, int y1);
  void makeHistogram(float *dispData, int width, int height, Pixels *spPixels, int spI);
  float getProbability(DispHistogram *hist2);
  void subtractHistogram(DispHistogram *cH2);
  void addHistogram(DispHistogram *cH2);
  void calcPDF();
  void print();
  float *pdf;
  int *histogram;
  int nrBins;
  int nrDataPoints;
  int nrPixelPoints;
  int filterSize;     // Size of the 1D gaussian filter. Should be an odd number and (filterSize-1)/2 < nrBins
  float filterSigma;  // The standard deviation of the gaussian filter

 private:
  void emptyHistogram();
  static float *gaussFilter1D(int size, float sigma);
  void smoothHistogram();
};

#endif
