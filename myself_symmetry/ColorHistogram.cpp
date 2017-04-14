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

#include "ColorHistogram.h"
#include "additional.h"
ColorHistogram::ColorHistogram(ColorHistogram *cH2, bool copyPDF) {
  // Makes a copy of the color histogram. NB: Does not copy the pdf array is copyPDF==false, only allocates memory!!
  nrBins = cH2->nrBins;
  nrBinsT = cH2->nrBinsT;
  colorModel = cH2->colorModel;
  nrChannels = cH2->nrChannels;
  nrDataPoints = cH2->nrDataPoints;
  filterSize = cH2->filterSize;
  filterSigma = cH2->filterSigma;

  histogram = new int[nrBinsT];
  for(int i=0; i<nrBinsT; i++)
    histogram[i] = cH2->histogram[i];
  pdf = new float[nrBinsT];
  if(copyPDF)
    for(int i=0; i<nrBinsT; i++)
      pdf[i] = cH2->pdf[i];
    
}

ColorHistogram::ColorHistogram(int nB, ColorModel cm) {
  // pow(nB, 3) should fit in an int
  filterSize = 5; 
  filterSigma = 1.0f;
  nrBins = nB;
  colorModel = cm;
  if(colorModel==CM_RGB || colorModel==CM_LAB) {
    nrBinsT = nrBins*nrBins*nrBins;
    nrChannels = 3; 
  }
  else { //CM_AB
    nrBinsT = nrBins*nrBins;
    nrChannels = 2; 
  }
  histogram = new int[nrBinsT];
  pdf = new float[nrBinsT];
  //signature = cvCreateMat(nrBinsT, nrChannels+1, CV_32FC1);
  emptyHistogram();
}

ColorHistogram::~ColorHistogram() {
  delete[] histogram;
  delete[] pdf;
  //cvReleaseMat(&signature);
}

void ColorHistogram::emptyHistogram() {
  for(int i=0; i<nrBinsT; i++)
    histogram[i] = 0.0;
  nrDataPoints = 0;
}

void ColorHistogram::makeHistogram(float *imgData, int width, int height, Pixels *spPixels, int spI) {
  if(nrDataPoints>0)
    emptyHistogram();
  
  int idx;
  float binWidth = 1.0f/nrBins;

  if(colorModel==CM_RGB) {
    int rI, gI, bI;
    for(unsigned int i=0; i<spPixels->pixels.size(); i++) {
      idx = 3*spPixels->pixels[i];
      rI = (int)(imgData[idx+2]/(binWidth+FLT_EPSILON));
      gI = (int)(imgData[idx+1]/(binWidth+FLT_EPSILON));
      bI = (int)(imgData[idx]/(binWidth+FLT_EPSILON));
      histogram[histI3(rI,gI,bI)]++;
      nrDataPoints++;
    }  
  }
  else if(colorModel==CM_LAB) {
    int yI, crI, cbI;
    for(unsigned int i=0; i<spPixels->pixels.size(); i++) {
      idx = 3*spPixels->pixels[i];
      yI  = (int)(imgData[idx]/(binWidth+FLT_EPSILON));
      crI = (int)(imgData[idx+1]/(binWidth+FLT_EPSILON));
      cbI = (int)(imgData[idx+2]/(binWidth+FLT_EPSILON));
      histogram[histI3(yI,crI,cbI)]++;
      nrDataPoints++;
    }  
  }
  else { // if(colorModel==CM_AB
    int crI, cbI;
    for(unsigned int i=0; i<spPixels->pixels.size(); i++) {
      idx = 3*spPixels->pixels[i];
      crI = (int)(imgData[idx+1]/(binWidth+FLT_EPSILON));
      cbI = (int)(imgData[idx+2]/(binWidth+FLT_EPSILON));
      histogram[histI2(crI,cbI)]++;
      nrDataPoints++;
    }  
  }

  smoothHistogram();
  calcPDF();
  //calcSignature();
}

void ColorHistogram::makeHistogram(float *imgData, int width, int height, int x0, int y0, int x1, int y1) {
  // Makes a histogram from the data in imgData that is inside the square defined by x0, y0, x1, y1
  // imgData needs to be a pointer to bgr image data.
  // TODO: smoothing of histogram for generalisation
  if(nrDataPoints>0)
    emptyHistogram();
  
  int i;
  float binWidth = 1.0f/nrBins;

  if(colorModel==CM_RGB) {
    int rI, gI, bI;
    for(int y=y0; y<y1; y++) {
      for(int x=x0; x<x1; x++) {
	i = 3*(y*width + x);
	rI = (int)(imgData[i+2]/(binWidth+FLT_EPSILON));
	gI = (int)(imgData[i+1]/(binWidth+FLT_EPSILON));
	bI = (int)(imgData[i]/(binWidth+FLT_EPSILON));
	histogram[histI3(rI,gI,bI)]++;
	nrDataPoints++;
      }
    } 
  }
  else if(colorModel==CM_LAB) {
    int yI, crI, cbI;
    for(int y=y0; y<y1; y++) {
      for(int x=x0; x<x1; x++) {
	i = 3*(y*width + x);
	yI = (int)(imgData[i]/(binWidth+FLT_EPSILON));
	crI = (int)(imgData[i+1]/(binWidth+FLT_EPSILON));
	cbI = (int)(imgData[i+2]/(binWidth+FLT_EPSILON));
	histogram[histI3(yI, crI,cbI)]++;
	nrDataPoints++;
      }
    } 
  }
  else { // if(colorModel==CM_AB
    int crI, cbI;
    for(int y=y0; y<y1; y++) {
      for(int x=x0; x<x1; x++) {
	i = 3*(y*width + x);
	crI = (int)(imgData[i+1]/(binWidth+FLT_EPSILON));
	cbI = (int)(imgData[i+2]/(binWidth+FLT_EPSILON));
	histogram[histI2(crI,cbI)]++;
	nrDataPoints++;
      }
    } 
  }

  smoothHistogram();
  calcPDF();
  //calcSignature();
}

void ColorHistogram::calcPDF() {
  for(int i=0; i<nrBinsT; i++)
    pdf[i] = (float)histogram[i]/nrDataPoints;
}

void ColorHistogram::calcPDFNorm0to1() {
  int maxV = -1;
  for(int i=0; i<nrBinsT; i++)
    if(histogram[i]>maxV)
      maxV = histogram[i];
  for(int i=0; i<nrBinsT; i++) {
    pdf[i] = (float)histogram[i]/(float)maxV;
  }
}

/*
void ColorHistogram::calcSignature() {
  // Calculate histogram signature
  int i;
  float *sign = (float *)signature->data.fl;

  if(colorModel==CM_RGB) {
    for(int rI=0; rI<nrBins; rI++) {
      for(int gI=0; gI<nrBins; gI++) {
	for(int bI=0; bI<nrBins; bI++) {
	  i = histI3(rI,gI,bI);
	  sign[4*i] = (float)histogram[i];
	  sign[4*i+1] = (float)rI;
	  sign[4*i+2] = (float)gI;
	  sign[4*i+3] = (float)bI;
	}
      }
    }
  }
  else if(colorModel==CM_LAB) {
    for(int yI=0; yI<nrBins; yI++) {
      for(int crI=0; crI<nrBins; crI++) {
	for(int cbI=0; cbI<nrBins; cbI++) {
	  i = histI3(yI,crI,cbI);
	  sign[4*i] = (float)histogram[i];
	  sign[4*i+1] = (float)yI;
	  sign[4*i+2] = (float)crI;
	  sign[4*i+3] = (float)cbI;
	}
      }
    }
  }
  else { // if(colorModel==CM_AB
    for(int crI=0; crI<nrBins; crI++) {
      for(int cbI=0; cbI<nrBins; cbI++) {
	i = histI2(crI,cbI);
	sign[3*i] = (float)histogram[i];
	sign[3*i+1] = (float)crI;
	sign[3*i+2] = (float)cbI;
      }
    }
  }
}
*/

void ColorHistogram::extendHistogram(uchar *imgData, int width, int height, int x0, int y0, int x1, int y1) {
  // Adds data in imgData that is inside the square defined by x0, y0, x1, y1 to the histogram
  // imgData needs to be a pointer to bgr image data.
  // TODO: smoothing of histogram for generalisation
  printf("NOT IMPLEMENTED\n");
  /*
  int rI, gI, bI;
  int i;
  float binWidth = 1.0f/nrBins;
  for(int y=y0; y<y1; y++) {
    for(int x=x0; x<x1; x++) {
      i = 3*(y*width + x);
      rI = (int)(imgData[i+2]/(binWidth+FLT_EPSILON));
      gI = (int)(imgData[i+1]/(binWidth+FLT_EPSILON));
      bI = (int)(imgData[i]/(binWidth+FLT_EPSILON));
      histogram[histI(rI,gI,bI)]++;
      nrDataPoints++;
    }
  } 
  calcPDF();
  calcSignature();
  */
}

void ColorHistogram::print() {
  if(colorModel==CM_RGB) {
    for(int rI=0; rI<nrBins; rI++) {
      for(int gI=0; gI<nrBins; gI++) {
	for(int bI=0; bI<nrBins; bI++) {
	  printf("(%d, %d, %d): %f\n", rI,gI,bI,pdf[histI3(rI,gI,bI)]);
	}
      }
    }
  }
  else if(colorModel==CM_LAB) {
    for(int yI=0; yI<nrBins; yI++) {
      for(int crI=0; crI<nrBins; crI++) {
	for(int cbI=0; cbI<nrBins; cbI++) {
	  printf("(%d %d, %d): %f\n", yI, crI,cbI,pdf[histI3(yI,crI,cbI)]);
	}
      }
    }
  }
  else { // colorModel==CM_AB
    for(int crI=0; crI<nrBins; crI++) {
      for(int cbI=0; cbI<nrBins; cbI++) {
	printf("%05d ", histogram[histI2(crI,cbI)]);
      }
      printf("\n");
    }
  }
}

float ColorHistogram::getProbability(float c1, float c2, float c3) {
  float binWidth = 1.0f/nrBins;
  if(colorModel==CM_RGB) {
    int rI = (int)(c1/(binWidth+FLT_EPSILON));
    int gI = (int)(c2/(binWidth+FLT_EPSILON));
    int bI = (int)(c3/(binWidth+FLT_EPSILON));
    return(pdf[histI3(rI,gI,bI)]);
  }
  else if(colorModel==CM_LAB) {
    int yI = (int)(c1/(binWidth+FLT_EPSILON));
    int crI = (int)(c2/(binWidth+FLT_EPSILON));
    int cbI = (int)(c3/(binWidth+FLT_EPSILON));
    return(pdf[histI3(yI, crI, cbI)]);
  }
  else { // colroModel==CM_AB
    int crI = (int)(c2/(binWidth+FLT_EPSILON));
    int cbI = (int)(c3/(binWidth+FLT_EPSILON));
    return(pdf[histI2(crI, cbI)]);
  }
}

float ColorHistogram::getProbability(ColorHistogram *h2) {
  // Compare the two histrograms. Use correlation. Correlations < 0 are set to 0
  if(colorModel==h2->colorModel || nrBins!=h2->nrBins) {
    float x = 0.0, x2 = 0.0;
    float y = 0.0, y2 = 0.0;
    for(int i=0; i<nrBinsT; i++) {
      x += pdf[i];
      y += h2->pdf[i];
      x2 += pdf[i]*pdf[i];
      y2 += h2->pdf[i]*h2->pdf[i];
    }
    float meanX = x/(nrBinsT);
    float meanY = y/(nrBinsT);
    float varX = x2 - x*x/(nrBinsT);
    float varY = y2 - y*y/(nrBinsT);
    float xy = 0.0;
    for(int i=0; i<nrBinsT; i++)
      xy += (pdf[i]-meanX)*(h2->pdf[i]-meanY);
    float corr = xy / sqrt(varX*varY);
    corr = corr>0?corr:0;
    corr = corr>1?1:corr;
    return(corr);
  }
  else {
    printf("ERROR: ColorHistogram::getProbability, histograms have different color models or different number of bins\n");
    exit(0);
    return(0);
  }
}


float ColorHistogram::getQuadraticDistance(ColorHistogram *h2) {
  printf("NOT IMPLEMENTED\n");
  /*
  float *aRGB = new float[nrBinsT*nrBinsT];
  float normA = sqrt(3)*nrBins;
  int i, i2;
  for(int rI=0; rI<nrBins; rI++) {
    for(int gI=0; gI<nrBins; gI++) {
      for(int bI=0; bI<nrBins; bI++) {
	i = histI(rI,gI,bI);
	for(int rI2=0; rI2<nrBins; rI2++) {
	  for(int gI2=0; gI2<nrBins; gI2++) {
	    for(int bI2=0; bI2<nrBins; bI2++) {
	      i2 = histI(rI2,gI2,bI2);
	      aRGB[i*nrBinsT + i2] = 1 - sqrt( (float)(rI-rI2)*(rI-rI2) + (float)(gI-gI2)*(gI-gI2) + (float)(bI-bI2)*(bI-bI2) ) / normA;
	    }
	  }
	}
      }
    }
  }

  float sumD = 0.0f, partSumD;
  for(i=0; i<nrBinsT; i++) {
    partSumD = 0.0f;
    for(i2=0; i2<nrBinsT; i2++) {
      partSumD += aRGB[i*nrBinsT+i2] * (pdf[i2] - h2->pdf[i2]);
    }
    sumD += (pdf[i] - h2->pdf[i]) * partSumD;
  }
  //printf("%f\n", sumD);

  delete[] aRGB;
  return(sqrt(sumD));
  */
  return(0.0);
}

/*
float ColorHistogram::getEMD(ColorHistogram *h2) {
  // Calculate the Earth Movers Distance. This includes the distance between bins and is therefore
  // more appropriate for color histograms, since bins are not independent. 
  // TODO: precaclulate the CvArrs to increase speed
  //cvReleaseMat(&signature);
  //signature = cvCreateMat(nrBinsT, nrChannels+1, CV_32FC1);

  if(colorModel==h2->colorModel || nrBins!=h2->nrBins) {
    float emd;
    if(colorModel==CM_RGB)
      emd = cvCalcEMD2(signature, h2->signature, CV_DIST_L2) / sqrt(3); // sqrt(3) maximum L2 measure in RGB
    else if(colorModel==CM_LAB)
      emd = cvCalcEMD2(signature, h2->signature, CV_DIST_L2) / sqrt(3); // sqrt(3) maximum L2 measure in RGB
    else // colorModel==CM_AB
      emd = cvCalcEMD2(signature, h2->signature, CV_DIST_L2) / sqrt(2); // sqrt(3) maximum L2 measure in RGB
    return(emd);
  }
  else{ 
    printf("ERROR: ColorHistogram::getEMD, histograms have different color models or different number of bins\n");
    exit(0);
    return(0);
  }
  //float emd = cvCalcEMD2(signature, signature2, CV_DIST_L2) / sqrt(3); // sqrt(3) maximum L2 measure in RGB
  //printf("emd: %f\n", emd);
  //cvReleaseMat(&signature2);
}
*/

float *ColorHistogram::gaussFilter1D(int size, float sigma) {
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

void ColorHistogram::smoothHistogram() {
  // Filter histogram with a Gaussian filter. Since this filter is separable,
  // it can be done by a series of 1D filters in all dimensions
  static float *filter = gaussFilter1D(filterSize, filterSigma);
  int filterR = (filterSize-1)/2;

  if(colorModel==CM_RGB || colorModel==CM_LAB){
    // 3D histogram
    int c1, c2, c3, f;
    float v, w;
    // Filter in all dimensions
    for(int channel=0; channel<nrChannels; channel++) {
      int *histogramF = new int[nrBinsT]; // To store the intermediate results
      for(c1=0; c1<nrBins; c1++) {
	for(c2=0; c2<nrBins; c2++) {
	  // Process the left/top/front border
	  for(c3=0; c3<filterR; c3++) { 
	    v = 0.0f; 
	    w = 0.0f;
	    for(f=-c3; f<=filterR; f++) {
	      w += filter[f+filterR];
	      if(channel==0)
		v += filter[f+filterR]*histogram[histI3(c1,c2,c3+f)];
	      if(channel==1)
		v += filter[f+filterR]*histogram[histI3(c1,c3+f,c2)];
	      else
		v += filter[f+filterR]*histogram[histI3(c3+f,c1,c2)];
	    }
	    if(channel==0)
	      histogramF[histI3(c1,c2,c3)] = (int)rint(v/w);
	    if(channel==1)
	      histogramF[histI3(c1,c3,c2)] = (int)rint(v/w);
	    else	  
	      histogramF[histI3(c3,c1,c2)] = (int)rint(v/w);
	  }
	  // Process the right/bottom/back border
	  for(c3=nrBins-filterR; c3<nrBins; c3++) { 
	    v = 0.0f; 
	    w = 0.0f;
	    for(f=-filterR; f<=(nrBins-1-c3); f++) {
	      w += filter[f+filterR];
	      if(channel==0)
		v += filter[f+filterR]*histogram[histI3(c1,c2,c3+f)];
	      if(channel==1)
		v += filter[f+filterR]*histogram[histI3(c1,c3+f,c2)];
	      else
		v += filter[f+filterR]*histogram[histI3(c3+f,c1,c2)];
	    }
	    if(channel==0)
	      histogramF[histI3(c1,c2,c3)] = (int)rint(v/w);
	    if(channel==1)
	      histogramF[histI3(c1,c3,c2)] = (int)rint(v/w);
	    else	  
	      histogramF[histI3(c3,c1,c2)] = (int)rint(v/w);
	  }
	  // Process the central part
	  for(c3=filterR; c3<nrBins-filterR; c3++) { 
	    v = 0.0f;
	    for(f=-filterR; f<=filterR; f++) {
	      if(channel==0)
		v += filter[f+filterR]*histogram[histI3(c1,c2,c3+f)];
	      if(channel==1)
		v += filter[f+filterR]*histogram[histI3(c1,c3+f,c2)];
	      else
		v += filter[f+filterR]*histogram[histI3(c3+f,c1,c2)];
	    }
	    if(channel==0)
	      histogramF[histI3(c1,c2,c3)] = (int)rint(v);
	    if(channel==1)
	      histogramF[histI3(c1,c3,c2)] = (int)rint(v);
	    else	  
	      histogramF[histI3(c3,c1,c2)] = (int)rint(v);
	  }
	} // for(c2=0; c2<nrBins; c2++) {
      } // for(c1=0; c1<nrBins; c1++) {
      delete[] histogram;
      histogram = histogramF;
    } // for(int channel=0; channel<nrChannels; channel++) {
  }
  else { // colorModel==AB
    // 2D histogram
    int c1, c2, f;
    float v, w;
    // Filter in all dimensions
    for(int channel=0; channel<nrChannels; channel++) {
      int *histogramF = new int[nrBinsT]; // To store the intermediate results
      for(c1=0; c1<nrBins; c1++) {
	// Process the left/top border
	for(c2=0; c2<filterR; c2++) { 
	  v = 0.0f; 
	  w = 0.0f;
	  for(f=-c2; f<=filterR; f++) {
	    w += filter[f+filterR];
	    if(channel==0)
	      v += filter[f+filterR]*histogram[histI2(c1,c2+f)];
	    else
	      v += filter[f+filterR]*histogram[histI2(c2+f,c1)];
	  }
	  if(channel==0)
	    histogramF[histI2(c1,c2)] = (int)rint(v/w);
	  else	  
	    histogramF[histI2(c2,c1)] = (int)rint(v/w);
	}
	// Process the right/bottom border
	for(c2=nrBins-filterR; c2<nrBins; c2++) { 
	  v = 0.0f; 
	  w = 0.0f;
	  for(f=-filterR; f<=(nrBins-1-c2); f++) {
	    w += filter[f+filterR];
	    if(channel==0)
	      v += filter[f+filterR]*histogram[histI2(c1,c2+f)];
	    else
	      v += filter[f+filterR]*histogram[histI2(c2+f,c1)];
	  }
	  if(channel==0)
	    histogramF[histI2(c1,c2)] = (int)rint(v/w);
	  else	  
	    histogramF[histI2(c2,c1)] = (int)rint(v/w);
	}
	// Process the central part
	for(c2=filterR; c2<nrBins-filterR; c2++) { 
	  v = 0.0f;
	  for(f=-filterR; f<=filterR; f++) {
	    if(channel==0)
	      v += filter[f+filterR]*histogram[histI2(c1,c2+f)];
	    else
	      v += filter[f+filterR]*histogram[histI2(c2+f,c1)];
	  }
	  if(channel==0)
	    histogramF[histI2(c1,c2)] = (int)rint(v);
	  else	  
	    histogramF[histI2(c2,c1)] = (int)rint(v);
	}
      } // for(c1=0; c1<nrBins; c1++) {
      delete[] histogram;
      histogram = histogramF;
    } // for(int channel=0; channel<nrChannels; channel++) {
  } // if(colorModel...)
  // Recalculate the number of data points, since this might have changes due to rounding differences
  nrDataPoints = 0;
  for(int i=0; i<nrBinsT; i++)
    nrDataPoints += histogram[i];
}

void ColorHistogram::subtractHistogram(ColorHistogram *cH2) {
  // Subtract the histogram values in cH2 from this and recalculate the PDF
  // this.histogram and cH2.histogram contain smoothed data, so no smoothing is necessary
  // NB: the algorithm assumes that this.histogram[i] >= cH2.histogram[i] !!
  for(int i=0; i<nrBinsT; i++) 
    histogram[i] -= cH2->histogram[i];
  nrDataPoints -= cH2->nrDataPoints;
}

void ColorHistogram::addHistogram(ColorHistogram *cH2) {
  for(int i=0; i<nrBinsT; i++)
    histogram[i] += cH2->histogram[i];
  nrDataPoints += cH2->nrDataPoints;
}

