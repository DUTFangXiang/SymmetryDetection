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

#include "DispHistogram.h"
#include "additional.h"
 
DispHistogram::DispHistogram(int nrB) {
  filterSize = 9; 
  filterSigma = 4.0f;
  nrBins = nrB;
  histogram = new int[nrBins];
  pdf = new float[nrBins];
  emptyHistogram();
}

DispHistogram::DispHistogram(DispHistogram *dH2, bool copyPDF) {
  nrBins = dH2->nrBins;
  nrDataPoints = dH2->nrDataPoints;
  nrPixelPoints = dH2->nrPixelPoints;
  filterSize = dH2->filterSize;
  filterSigma = dH2->filterSigma;

  histogram = new int[nrBins];
  for(int i=0; i<nrBins; i++)
    histogram[i] = dH2->histogram[i];
  pdf = new float[nrBins];
  if(copyPDF)
    for(int i=0; i<nrBins; i++)
      pdf[i] = dH2->pdf[i];
}

DispHistogram::~DispHistogram() {
  delete[] histogram;
  delete[] pdf;
}

void DispHistogram::makeHistogram(float *dispData, int width, int height, int x0, int y0, int x1, int y1) {
  if(nrDataPoints>0)
    emptyHistogram();
  
  int i, dI;
  float binWidth = 1.0f/nrBins;

  for(int y=y0; y<y1; y++) {
    for(int x=x0; x<x1; x++) {
      i = y*width + x;
      if(dispData[i]>=0) {
	dI = (int)(dispData[i]/(binWidth+FLT_EPSILON));
	histogram[dI]++;
	nrDataPoints++;
      }
      nrPixelPoints++;
    } 
  }
  smoothHistogram();
  calcPDF();
  //calcSignature();
}

void DispHistogram::makeHistogram(float *dispData, int width, int height, Pixels *spPixels, int spI) {
  if(nrDataPoints>0)
    emptyHistogram();
  
  int idx, dI;
  float binWidth = 1.0f/nrBins;

  for(unsigned int i=0; i<spPixels->pixels.size(); i++) {
    idx = spPixels->pixels[i];
    if(dispData[idx]>=0) {
      dI = (int)(dispData[idx]/(binWidth+FLT_EPSILON));
      histogram[dI]++;
      nrDataPoints++;
    }
    nrPixelPoints++;		
  }
  smoothHistogram();
  calcPDF();
}

float DispHistogram::getProbability(DispHistogram *h2) {
  // Compare the two histrograms. Use correlation. Correlations < 0 are set to 0
  float x = 0.0, x2 = 0.0;
  float y = 0.0, y2 = 0.0;
  for(int i=0; i<nrBins; i++) {
    x += pdf[i];
    y += h2->pdf[i];
    x2 += pdf[i]*pdf[i];
    y2 += h2->pdf[i]*h2->pdf[i];
  }
  float meanX = x/(nrBins);
  float meanY = y/(nrBins);
  float varX = x2 - x*x/(nrBins);
  float varY = y2 - y*y/(nrBins);
  float xy = 0.0;
  for(int i=0; i<nrBins; i++)
    xy += (pdf[i]-meanX)*(h2->pdf[i]-meanY);
  float corr = xy / sqrt(varX*varY);
  corr = corr>0?corr:0;
  corr = corr>1?1:corr;
  return(corr);
}

void DispHistogram::subtractHistogram(DispHistogram *cH2) {
  for(int i=0; i<nrBins; i++) 
    histogram[i] -= cH2->histogram[i];
  nrDataPoints -= cH2->nrDataPoints;
  nrPixelPoints -= cH2->nrPixelPoints;
}

void DispHistogram::addHistogram(DispHistogram *cH2) {
  for(int i=0; i<nrBins; i++)
    histogram[i] += cH2->histogram[i];
  nrDataPoints += cH2->nrDataPoints;
  nrPixelPoints += cH2->nrPixelPoints;
}

void DispHistogram::calcPDF() {
  for(int i=0; i<nrBins; i++)
    pdf[i] = (float)histogram[i]/nrDataPoints;
}

void DispHistogram::emptyHistogram() {
  for(int i=0; i<nrBins; i++)
    histogram[i] = 0.0;
  nrDataPoints = 0;
  nrPixelPoints = 0;
}

float *DispHistogram::gaussFilter1D(int size, float sigma) {
  float *y = new float[size];
  float m = ((float)size-1)/2;
  float sumY = 0.0f;
  for(int i=0; i<size; i++) {
    y[i] = (1.0f/(sigma*sqrt(2*M_PI))) * exp(-(i-m)*(i-m)/(2*sigma*sigma));
    sumY += y[i];
  }
  for(int i=0; i<size; i++)
    y[i] /= sumY;
  
  return(y);
}

void DispHistogram::smoothHistogram() {
  // Filter histogram with a Gaussian filter. 
  static float *filter = gaussFilter1D(filterSize, filterSigma);
  int filterR = (filterSize-1)/2;

  int d1, f;
  float v, w;
  
  int *histogramF = new int[nrBins]; // To store the intermediate results
  // Process the left border
  for(d1=0; d1<filterR; d1++) { 
    v = 0.0f; 
    w = 0.0f;
    for(f=-d1; f<=filterR; f++) {
      w += filter[f+filterR];
      v += filter[f+filterR]*histogram[d1+f];
    }
    histogramF[d1] = (int)rint(v/w);
  }
  // Process the right border
  for(d1=nrBins-filterR; d1<nrBins; d1++) { 
    v = 0.0f; 
    w = 0.0f;
    for(f=-filterR; f<=(nrBins-1-d1); f++) {
      w += filter[f+filterR];
      v += filter[f+filterR]*histogram[d1+f];
    }
    histogramF[d1] = (int)rint(v/w);
  }
  // Process the central part
  for(d1=filterR; d1<nrBins-filterR; d1++) { 
    v = 0.0f;
    for(f=-filterR; f<=filterR; f++) {
      v += filter[f+filterR]*histogram[d1+f];
    }
    histogramF[d1] = (int)rint(v/w);
  }
  delete[] histogram;
  histogram = histogramF;

  // Recalculate the number of data points, since this might have changes due to rounding differences
  nrDataPoints = 0;
  for(int i=0; i<nrBins; i++)
    nrDataPoints += histogram[i];
}

void DispHistogram::print() {
  for(int i=0; i<nrBins; i++)
    printf("%04d ", histogram[i]);
  printf("\n");
}
