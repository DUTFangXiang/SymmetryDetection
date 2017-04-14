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

#ifndef COLORHISTOGRAM_H
#define COLORHISTOGRAM_H

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

enum ColorModel {CM_RGB, CM_LAB, CM_AB};

//! Class containing the color histogram and methods to manipulate the histogram and calculate probabilities
class ColorHistogram {
public:
  ColorHistogram(int nB, ColorModel cm);
  ColorHistogram(ColorHistogram *colorHist, bool copyPDF);
  ~ColorHistogram();
  void makeHistogram(float *imgData, int width, int height, int x0, int y0, int x1, int y1);
  void makeHistogram(float *imgData, int width, int height, Pixels *pixels, int spI);
  void extendHistogram(uchar *imgData, int width, int height, int x0, int y0, int x1, int y1);
  void print();
  float getProbability(float r, float g, float b);
  float getProbability(ColorHistogram *hist2);
  float getQuadraticDistance(ColorHistogram *hist2);
  //float getEMD(ColorHistogram *hist2);
  void subtractHistogram(ColorHistogram *cH2);
  void addHistogram(ColorHistogram *cH2);
  void calcPDF();

  float *pdf;
  int *histogram;
  //CvMat *signature;
  int nrBins;
  ColorModel colorModel;
  int nrBinsT; // total nr of bins
  int nrChannels;
  int nrDataPoints;
  int filterSize;     // Size of the 1D gaussian filter. Should be an odd number and (filterSize-1)/2 < nrBins
  float filterSigma;  // The standard deviation of the gaussian filter
 private:
  void emptyHistogram();
  void calcPDFNorm0to1();
  //void calcSignature();
  int histI3(int c1I, int c2I, int c3I) { return(c1I*nrBins*nrBins + c2I*nrBins + c3I); };
  int histI2(int c1I, int c2I) { return(c1I*nrBins + c2I); };
  static float *gaussFilter1D(int size, float sigma);
  void smoothHistogram();

};

#endif
