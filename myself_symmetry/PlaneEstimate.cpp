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

#include "PlaneEstimate.h"

/*
void PlaneEstimate::findTablePlane(IplImage* pImg, sSurface &surface) {

  int width = pImg->width;
  int height = pImg->height;
  
  float* lDisps = new float[width*height];
  int* wCoords = new int[width*height];
  int* hCoords = new int[width*height];
  
  int counter = 0;
  for(int h=0; h<height; ++h) {
    for(int w=0; w<width; ++w) {
      if(CV_IMAGE_ELEM(pImg, unsigned char, h, w) != (unsigned char)255) {
	lDisps[counter] = CV_IMAGE_ELEM(pImg, unsigned char, h, w );
        wCoords[counter] = w;
        hCoords[counter++] = h;
      }
    }
  }

  findPlane(lDisps, wCoords, hCoords, counter, height, surface, 1000);


  delete [] lDisps;
  delete [] wCoords;
  delete [] hCoords;
  
}*/

void PlaneEstimate::findTablePlane(float* pImg, sSurface &surface) {

  int width = sp->width;
  int height = sp->height;
  
  float* lDisps = new float[width*height];
  int* wCoords = new int[width*height];
  int* hCoords = new int[width*height];
  
  int counter = 0;
  for(int h=0; h<height; ++h) {
    for(int w=0; w<width; ++w) {
      if(pImg[h*width+w] != -1) {
	lDisps[counter] = pImg[h*width+w];
        wCoords[counter] = w;
        hCoords[counter++] = h;
      }
    }
  }

  findPlanef(lDisps, wCoords, hCoords, counter, height, surface, 1000,0.002);


  delete [] lDisps;
  delete [] wCoords;
  delete [] hCoords;
  
}

/*
void PlaneEstimate::findSuperPixelPlane(IplImage *pImg, sSurface* surfaces) {

  int width = pImg->width;
  int height = pImg->height;

  int averageNrPixelsPerSP = round(width*height/sp->nrSPixels);
  
  vector<float*> lDisps;
  vector<int*> wCoords;
  vector<int*> hCoords;
  int* counters = new int[sp->nrSPixels];
  int* maxval = new int[sp->nrSPixels];

  for(int i=0; i<sp->nrSPixels; ++i) {
    // Allocate 10 times the average super pixel size
    lDisps.push_back(new float[10*averageNrPixelsPerSP]);
    wCoords.push_back(new int[10*averageNrPixelsPerSP]);
    hCoords.push_back(new int[10*averageNrPixelsPerSP]);
    counters[i] = 0;
    maxval[i] = 10*averageNrPixelsPerSP;
  }
  
  for(int h=0; h<height; ++h) {
    for(int w=0; w<width; ++w) {
      if(CV_IMAGE_ELEM(pImg, unsigned char, h, w) != (unsigned char)255) {
        int label = sp->sPixels[h*width+w];

        // If more space is needed, fix
        if(counters[label] == maxval[label]) {
          float* tmpD = new float[2*maxval[label]];
          int* tmpW = new int[2*maxval[label]];
          int* tmpH = new int[2*maxval[label]];
          memcpy(tmpD, lDisps[label],maxval[label]*sizeof(float));
          memcpy(tmpW, wCoords[label],maxval[label]*sizeof(int));
          memcpy(tmpH, hCoords[label],maxval[label]*sizeof(int));
          float* td = lDisps[label];
          int* tw = wCoords[label];
          int* th = hCoords[label];
          lDisps[label] = tmpD;
          wCoords[label] = tmpW;
          hCoords[label] = tmpH;
          delete [] td;
          delete [] tw;
          delete [] th;
        }
        
	(lDisps[label])[counters[label]] = CV_IMAGE_ELEM(pImg, unsigned char, h, w );
        (wCoords[label])[counters[label]] = w;
        (hCoords[label])[counters[label]] = h;
        ++counters[label];
      }
    }
  }

  struct timeval begin, end;
  for(int i=0; i<sp->nrSPixels; ++i) {
    //cout << "Disp points in sp " << i << ": " << counters[i] << endl;
    if(counters[i] > 10) {
      gettimeofday(&begin, NULL);
      findPlane(lDisps[i], wCoords[i], hCoords[i], counters[i],
                height, surfaces[i],100);
      gettimeofday(&end, NULL); 
      //cout << "    findTablePlane: " << 1000*(end.tv_sec-begin.tv_sec) + (end.tv_usec-begin.tv_usec)/1000 << " ms" << endl;
    } else {
      surfaces[i].alpha = 0;
      surfaces[i].beta = 0;
      surfaces[i].disp = 0;
    }
  }
  

  for(int i=0; i<sp->nrSPixels; ++i) {
    delete [] lDisps[i];
    delete [] wCoords[i];
    delete [] hCoords[i];
  }
  delete [] counters;
  delete [] maxval;
  
}
*/

void PlaneEstimate::findSuperPixelPlane(float *pImg, sSurface* surfaces) {

  int width = sp->width;
  int height = sp->height;

  int averageNrPixelsPerSP = round(width*height/sp->nrSPixels);
  
  vector<float*> lDisps;
  vector<int*> wCoords;
  vector<int*> hCoords;
  int* counters = new int[sp->nrSPixels];
  int* maxval = new int[sp->nrSPixels];

  for(int i=0; i<sp->nrSPixels; ++i) {
    // Allocate 10 times the average super pixel size
    lDisps.push_back(new float[10*averageNrPixelsPerSP]);
    wCoords.push_back(new int[10*averageNrPixelsPerSP]);
    hCoords.push_back(new int[10*averageNrPixelsPerSP]);
    counters[i] = 0;
    maxval[i] = 10*averageNrPixelsPerSP;
  }
  
  for(int h=0; h<height; ++h) {
    for(int w=0; w<width; ++w) {
      if(pImg[h*width+w] != -1) {
        int label = sp->sPixels[h*width+w];

        // If more space is needed, fix
        if(counters[label] == maxval[label]) {
          float* tmpD = new float[2*maxval[label]];
          int* tmpW = new int[2*maxval[label]];
          int* tmpH = new int[2*maxval[label]];
          memcpy(tmpD, lDisps[label],maxval[label]*sizeof(float));
          memcpy(tmpW, wCoords[label],maxval[label]*sizeof(int));
          memcpy(tmpH, hCoords[label],maxval[label]*sizeof(int));
          float* td = lDisps[label];
          int* tw = wCoords[label];
          int* th = hCoords[label];
          lDisps[label] = tmpD;
          wCoords[label] = tmpW;
          hCoords[label] = tmpH;
          delete [] td;
          delete [] tw;
          delete [] th;
        }
        
	(lDisps[label])[counters[label]] = pImg[h*width+w];
        (wCoords[label])[counters[label]] = w;
        (hCoords[label])[counters[label]] = h;
        ++counters[label];
      }
    }
  }

  struct timeval begin, end;
  for(int i=0; i<sp->nrSPixels; ++i) {
    //cout << "Disp points in sp " << i << ": " << counters[i] << endl;
    if(counters[i] > 10) {
      gettimeofday(&begin, NULL);
      findPlanef(lDisps[i], wCoords[i], hCoords[i], counters[i],
                 height, surfaces[i],100,0.0);
      gettimeofday(&end, NULL); 
      //cout << "    findTablePlane: " << 1000*(end.tv_sec-begin.tv_sec) + (end.tv_usec-begin.tv_usec)/1000 << " ms" << endl;
    } else {
      surfaces[i].alpha = 0;
      surfaces[i].beta = 0;
      surfaces[i].disp = 0;
    }
  }
  

  for(int i=0; i<sp->nrSPixels; ++i) {
    delete [] lDisps[i];
    delete [] wCoords[i];
    delete [] hCoords[i];
  }
  delete [] counters;
  delete [] maxval;
  
}

/*
void PlaneEstimate::findPlane(float* pDisps, int* wCoords, int* hCoords, int pNoPoints, int height, sSurface& surface, int randIter = 1000) {

  const int drange = 64;
  float minbeta = 0.0;
  
  int *hist = new int[height*drange];
  int *totals = new int[height];
  int *validh = new int[height];

  int counter = 0;
  int h;
  float d;
  // Search through all points
  for(int i=0; i<pNoPoints-1; ++i) {
    // Select h value
    h = hCoords[i];
    validh[counter] = h;
    int *histy = &hist[counter*drange];
    for(int j=0; j<drange; ++j) {
      histy[j] = 0;
    }
    // Loop until next h value
    int tmp=i;
    do {
      d = pDisps[i++];
      histy[(int)d]++;
    } while(hCoords[i-1] == hCoords[i]);
    
    for (int k=0;k<drange-2;k++)
      histy[k] = histy[k] + 2*histy[k+1] + histy[k+2];
    for (int k=drange-1;k>1;k--)
      histy[k] = histy[k] + 2*histy[k-1] + histy[k-2];
    totals[counter] = 0;
    for (int k=0;k<drange;k++) 
      totals[counter] += histy[k];

    counter++;
  }
  
  // Find best line using random sampling
  height = counter;
  float maxwei = 0.0f; 
  float alpha = 0.0f; 
  float beta = 0.0f; 
  float disp = drange/2;
  for (int l=0;l<randIter;l++) {
    int idx1 = rand()%height;
    int idx2 = rand()%height;
    while (idx1==idx2)
      idx2 = rand()%height;
    if (!totals[idx1] || !totals[idx2])
      continue;
    int cnt1 = rand()%totals[idx1];
    int cnt2 = rand()%totals[idx2];
    int disp1 = 0, disp2 = 0;
    for (int sum1=0;sum1<cnt1;disp1++) 
      sum1 += hist[idx1*drange+disp1];
    for (int sum2=0;sum2<cnt2;disp2++) 
      sum2 += hist[idx2*drange+disp2];
    disp1--;
    disp2--;
    float dgra = (float)(disp2 - disp1) / (validh[idx2] - validh[idx1]);
    float dzer = disp2 - dgra*validh[idx2];
    float sumwei = 0.0f;
    for (int y=0;y<height;y++) {
      for (int dd=-3;dd<=3;dd++) {
	int d = (int)(dgra*validh[y] + dzer + 0.5f) + dd;
	if (d<0 || d>=drange) 
	  continue;
	float er = d - (dgra*validh[y] + dzer);
	sumwei += hist[y*drange + d] / (4.0f + er*er);
      }
    }
    if (sumwei>maxwei && dgra>minbeta) {
      maxwei = sumwei;
      beta = dgra;
      disp = dzer;
    }
  }

  // Improve line (depends only on y) using m-estimator
  for (int l=0;l<3;l++) {
    float syy = 0.0, sy1 = 0.0f, s11 = 0.0f;
    float sdy = 0.0, sd1 = 0.0f;
    for (int yt=0;yt<height;yt++) {
      int y = validh[yt];
      for (int dd=-3;dd<=3;dd++) {
	int d = (int)(beta*y + disp + 0.5f) + dd;
	if (d<0 || d>=drange) 
	  continue;
	float er = d - (beta*y + disp);
	float w = hist[yt*drange + d] / (4.0f + er*er);
	syy += w*y*y;
	sy1 += w*y;
	s11 += w;
	sdy += w*d*y;
	sd1 += w*d;
      }
    }
    float det = syy*s11 - sy1*sy1;
    beta = s11*sdy - sy1*sd1;
    disp = syy*sd1 - sy1*sdy;
    if (det!=0.0f) {
      beta /= det;
      disp /= det;
    }
  }
  disp += 0.5f;

  // Improve plane (depends on both x and y) using m-estimator
  for (int l=0;l<3;l++) {
    float sxx = 0.0, sx1 = 0.0f, s11 = 0.0f;
    float sdx = 0.0, sd1 = 0.0f;

    for(int i=0; i<pNoPoints; ++i) {
      int y = hCoords[i];
      int x = wCoords[i];
      float pd = pDisps[i];
      
    
      //for (int y=0;y<height;y++) {
      //for (int x=0;x<width;x++) {
      if (pd>=0.0f) {
        float d = pd - beta*y;
        float er = d - (alpha*x + disp);
        float w = 1.0f / (1.0f + er*er);
        sxx += w*x*x;
        sx1 += w*x;
        s11 += w;
        sdx += w*d*x;
        sd1 += w*d;
      }
      //}
    }
    float det = sxx*s11 - sx1*sx1;
    alpha = s11*sdx - sx1*sd1;
    disp = sxx*sd1 - sx1*sdx;
    if (det!=0.0f) {
      alpha /= det;
      disp /= det;
    }
  }

  // Compute variance of d
  int num = 0;
  float vard = 0.0f;
  for (int i=0; i<pNoPoints; ++i) {
    //for (int y=0;y<height;y++) {
    //for (int x=0;x<width;x++) {
    //int i = y*width + x;
    float d = pDisps[i];
    int x = wCoords[i];
    int y = hCoords[i];
    if (d>=0.0f && d<drange) {
      float er = alpha*x + beta*y + disp - d;
      if (er*er<4*surface.spread_d) {
        vard += er*er;
        num++;
      }
    }
    //}
  }
  surface.spread_d = vard / num;
  surface.alpha = alpha;
  surface.beta = beta;
  surface.disp = disp;
  delete [] hist;
  delete [] totals;

  //std::cout << surface.spread_d << std::endl
  //            << surface.alpha << std::endl
  //            << surface.beta << std::endl
  //            << surface.disp << std::endl;
  
}*/

void PlaneEstimate::findPlanef(float* pDisps, int* wCoords, int* hCoords,
                               int pNoPoints, int height, sSurface& surface, int randIter = 1000,
							   float minbeta = 0.0) {

  const int drange = 64;
	// beta needs to be larger than a certain value to ensure horizontal plane
  
  float* lDisps = new float[pNoPoints];
  for(int i=0; i<pNoPoints; ++i) {
    if(pDisps[i] >= 1)
      lDisps[i] = drange-1;
    else
      lDisps[i] = (int)floor(pDisps[i]*drange);
  }
  
  int *hist = new int[height*drange];
  int *totals = new int[height];
  int *validh = new int[height];

  int counter = 0;
  int h;
  float d;
  // Search through all points
  for(int i=0; i<pNoPoints-1; i++) {
    // Select h value
    h = hCoords[i];
    validh[counter] = h;
    int *histy = &hist[counter*drange];
    for(int j=0; j<drange; ++j) {
      histy[j] = 0;
    }
    // Loop until next h value
    do {
      d = lDisps[i++];
      histy[(int)d]++;
    } while(hCoords[i-1] == hCoords[i]);
    
    for (int k=0;k<drange-2;k++)
      histy[k] = histy[k] + 2*histy[k+1] + histy[k+2];
    for (int k=drange-1;k>1;k--)
      histy[k] = histy[k] + 2*histy[k-1] + histy[k-2];
    totals[counter] = 0;
    for (int k=0;k<drange;k++) 
      totals[counter] += histy[k];

    counter++;
  }

  
  // Find best line using random sampling
  height = counter;
  float maxwei = 0.0f; 
  float alpha = 0.0f; 
  float beta = 0.0f; 
  float disp = drange/2;
  for (int l=0;l<randIter;l++) {
    int idx1 = rand()%height;
    int idx2 = rand()%height;
    while (idx1==idx2)
      idx2 = rand()%height;
    if (!totals[idx1] || !totals[idx2])
      continue;
    int cnt1 = rand()%totals[idx1];
    int cnt2 = rand()%totals[idx2];
    int disp1 = 0, disp2 = 0;
    for (int sum1=0;sum1<cnt1;disp1++) 
      sum1 += hist[idx1*drange+disp1];
    for (int sum2=0;sum2<cnt2;disp2++) 
      sum2 += hist[idx2*drange+disp2];
    disp1--;
    disp2--;
    float dgra = (float)(disp2 - disp1) / (validh[idx2] - validh[idx1]);
    float dzer = disp2 - dgra*validh[idx2];
    float sumwei = 0.0f;
    for (int y=0;y<height;y++) {
      for (int dd=-3;dd<=3;dd++) {
	int d = (int)(dgra*validh[y] + dzer + 0.5f) + dd;
	if (d<0 || d>=drange) 
	  continue;
	float er = d - (dgra*validh[y] + dzer);
	sumwei += hist[y*drange + d] / (4.0f + er*er);
      }
    }
	if (sumwei>maxwei && (dgra/drange>minbeta || minbeta == 0)) {
      maxwei = sumwei;
      beta = dgra;
      disp = dzer;
    }
  }
	
//std::cout << "first beta: " << beta << endl;

  // Improve line (depends only on y) using m-estimator
  for (int l=0;l<3;l++) {
    float syy = 0.0, sy1 = 0.0f, s11 = 0.0f;
    float sdy = 0.0, sd1 = 0.0f;
    for (int yt=0;yt<height;yt++) {
      int y = validh[yt];
      for (int dd=-3;dd<=3;dd++) {
	int d = (int)(beta*y + disp + 0.5f) + dd;
	if (d<0 || d>=drange) 
	  continue;
	float er = d - (beta*y + disp);
	float w = hist[yt*drange + d] / (4.0f + er*er);
	syy += w*y*y;
	sy1 += w*y;
	s11 += w;
	sdy += w*d*y;
	sd1 += w*d;
      }
    }
    float det = syy*s11 - sy1*sy1;
    beta = s11*sdy - sy1*sd1;
    disp = syy*sd1 - sy1*sdy;
    if (det!=0.0f) {
      beta /= det;
      disp /= det;
    }
  }
  disp += 0.5f;

  // Improve plane (depends on both x and y) using m-estimator
  for (int l=0;l<3;l++) {
    float sxx = 0.0, sx1 = 0.0f, s11 = 0.0f;
    float sdx = 0.0, sd1 = 0.0f;

    for(int i=0; i<pNoPoints; ++i) {
      int y = hCoords[i];
      int x = wCoords[i];
      float pd = lDisps[i];
      
    
      //for (int y=0;y<height;y++) {
      //for (int x=0;x<width;x++) {
      if (pd>=0.0f) {
        float d = pd - beta*y;
        float er = d - (alpha*x + disp);
        float w = 1.0f / (1.0f + er*er);
        sxx += w*x*x;
        sx1 += w*x;
        s11 += w;
        sdx += w*d*x;
        sd1 += w*d;
      }
      //}
    }
    float det = sxx*s11 - sx1*sx1;
    alpha = s11*sdx - sx1*sd1;
    disp = sxx*sd1 - sx1*sdx;
    if (det!=0.0f) {
      alpha /= det;
      disp /= det;
    }
  }

  // Compute variance of d
  int num = 0;
  float vard = 0.0f;
  for (int i=0; i<pNoPoints; ++i) {
    //for (int y=0;y<height;y++) {
    //for (int x=0;x<width;x++) {
    //int i = y*width + x;
    float d = lDisps[i];
    int x = wCoords[i];
    int y = hCoords[i];
    if (d>=0.0f && d<drange) {
      float er = alpha*x + beta*y + disp - d;
      if (er*er<4*surface.spread_d) {
        vard += er*er;
        num++;
      }
    }
    //}
  }
  int normfactor = drange;
	
/*if(beta/normfactor < minbeta)	{
  surface.spread_d = -1;
  surface.alpha = 0;
  surface.beta = 0;
  surface.disp = 0;	
} else*/ {
  surface.spread_d = vard / num;
  surface.alpha = alpha/normfactor;
  surface.beta = beta/normfactor;
  surface.disp = disp/normfactor;
}
  delete [] hist;
  delete [] totals;
  delete [] lDisps;
  delete[] validh;
  /*std::cout << surface.spread_d << std::endl
              << surface.alpha << std::endl
              << surface.beta << std::endl
              << surface.disp << std::endl;*/
  
}

void PlaneEstimate::findPlane(IplImage* pImg, sSurface& surface, int randIter = 1000) {

  int width = pImg->width;
  int height = pImg->height;
  
  int drange = 64;
  float *dimd = new float[width*height];
  float minbeta = 0.08;
  
  // Find dominating disparity for each y-value
  int *hist = new int[height*drange];
  int *totals = new int[height];
  for (int y=0;y<height;y++) {
    int *histy = &hist[y*drange];
    for (int i=0;i<drange;i++)
      histy[i] = 0;
    for (int x=0;x<width;x++) {
      uchar td = CV_IMAGE_ELEM(pImg, unsigned char, y, x);
      dimd[y*width+x] = td;//CV_IMAGE_ELEM(pImg, unsigned char, y, x);
      int d = (int)dimd[y*width + x];
      if (d>=0 && d<drange)
        histy[d] ++;
    }
    for (int i=0;i<drange-2;i++)
      histy[i] = histy[i] + 2*histy[i+1] + histy[i+2];
    for (int i=drange-1;i>1;i--)
      histy[i] = histy[i] + 2*histy[i-1] + histy[i-2];
    totals[y] = 0;
    for (int i=0;i<drange;i++)
      totals[y] += histy[i];
  }
  // Find best line using random sampling
  float maxwei = 0.0f;
  float alpha = 0.0f;
  float beta = 0.0f;
  float disp = drange/2;
  for (int l=0;l<randIter;l++) {
    int idx1 = rand()%height;
    int idx2 = rand()%height;
    while (idx1==idx2)
      idx2 = rand()%height;
    if (!totals[idx1] || !totals[idx2])
      continue;
    int cnt1 = rand()%totals[idx1];
    int cnt2 = rand()%totals[idx2];
    int disp1 = 0, disp2 = 0;
    for (int sum1=0;sum1<cnt1;disp1++)
      sum1 += hist[idx1*drange+disp1];
    for (int sum2=0;sum2<cnt2;disp2++)
      sum2 += hist[idx2*drange+disp2];
    disp1--;
    disp2--;
    float dgra = (float)(disp2 - disp1) / (idx2 - idx1);
    float dzer = disp2 - dgra*idx2;
    float sumwei = 0.0f;
    for (int y=0;y<height;y++) {
      for (int dd=-3;dd<=3;dd++) {
        int d = (int)(dgra*y + dzer + 0.5f) + dd;
        if (d<0 || d>=drange)
          continue;
        float er = d - (dgra*y + dzer);
        sumwei += hist[y*drange + d] / (4.0f + er*er);
      }
    }
    if (sumwei>maxwei && dgra>minbeta) {
      maxwei = sumwei;
      beta = dgra;
      disp = dzer;
    }
  }
  // Improve line (depends only on y) using m-estimator
  for (int l=0;l<3;l++) {
    float syy = 0.0, sy1 = 0.0f, s11 = 0.0f;
    float sdy = 0.0, sd1 = 0.0f;
    for (int y=0;y<height;y++) {
      for (int dd=-3;dd<=3;dd++) {
        int d = (int)(beta*y + disp + 0.5f) + dd;
        if (d<0 || d>=drange)
          continue;
        float er = d - (beta*y + disp);
        float w = hist[y*drange + d] / (4.0f + er*er);
        syy += w*y*y;
        sy1 += w*y;
        s11 += w;
        sdy += w*d*y;
        sd1 += w*d;
      }
    }
    float det = syy*s11 - sy1*sy1;
    beta = s11*sdy - sy1*sd1;
    disp = syy*sd1 - sy1*sdy;
    if (det!=0.0f) {
      beta /= det;
      disp /= det;
    }
  }
  disp += 0.5f;
  // Improve plane (depends on both x and y) using m-estimator
  for (int l=0;l<3;l++) {
    float sxx = 0.0, sx1 = 0.0f, s11 = 0.0f;
    float sdx = 0.0, sd1 = 0.0f;
    for (int y=0;y<height;y++) {
      for (int x=0;x<width;x++) {
        if (dimd[y*width+x]>=0.0f) {
          float d = dimd[y*width+x] - beta*y;
          float er = d - (alpha*x + disp);
          float w = 1.0f / (1.0f + er*er);
          sxx += w*x*x;
          sx1 += w*x;
          s11 += w;
          sdx += w*d*x;
          sd1 += w*d;
        }
      }
    }
    float det = sxx*s11 - sx1*sx1;
    alpha = s11*sdx - sx1*sd1;
    disp = sxx*sd1 - sx1*sdx;
    if (det!=0.0f) {
      alpha /= det;
      disp /= det;
    }
  }
  int num = 0;
  float vard = 0.0f;
  for (int y=0;y<height;y++) {
    for (int x=0;x<width;x++) {
      int i = y*width + x;
      float d = dimd[i];
      if (d>=0.0f && d<drange) {
        float er = alpha*x + beta*y + disp - d;
        if (er*er<4*surface.spread_d) {
          vard += er*er;
          num++;
        }
      }
    }
  }
  surface.spread_d = vard / num;
  surface.alpha = alpha;
  surface.beta = beta;
  surface.disp = disp;

  std::cout << "Original method: " << std::endl
            << "alpha: " << alpha
            << " beta: " << beta
            << " disp: " << disp << std::endl;
  
  
  delete [] hist;
  delete [] totals;
  delete [] dimd;
}
