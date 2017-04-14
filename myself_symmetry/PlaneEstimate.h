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

#ifndef PLANEESTIMATE_H
#define PLANEESTIMATE_H

#include "myTypes_sp.h"
#include "cv.h"
#include "SuperPixels.h"

//! Class to estimate the dominant plane and the plane of each individual super pixel
class PlaneEstimate {

  SuperPixels* sp;
  
public:
  PlaneEstimate(SuperPixels* pSp) :
    sp(pSp)
    {}
  

  void findSuperPixelPlane(IplImage* pImg, sSurface* surface);
  void findTablePlane(IplImage* pImg, sSurface &surface);


  void findSuperPixelPlane(float* pImg, sSurface* surface);
  void findTablePlane(float* pImg, sSurface &surface);


  void findPlane(float* pDisps, int* wCoords, int* hCoords,
                 int pNoPoints, int height, sSurface& surface, int randIter);
  void findPlanef(float* pDisps, int* wCoords, int* hCoords,
                  int pNoPoints, int height, sSurface& surface, int randIter,float minbeta);
  
  void findPlane(IplImage* pImg, sSurface& surface, int randIter);

};

#endif
