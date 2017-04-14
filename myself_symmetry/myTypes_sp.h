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

#ifndef MYTYPES_SP_H
#define MYTYPES_SP_H

#include "myTypes2.h"
#include "ColorHistogram.h"
#include "DispHistogram.h"
//#include "SuperPixels.h"

class SuperPixels;

//! Struct for the position of a pixel
struct Pixel {
  int x,y;
};

//! Struct storing surface-plane information
struct sSurface {
  float alpha;
  float beta;
  float disp;
  float spread_d;
};

//! Struct storing the link between two labels
struct LabelLink{
  int labelA;
  int labelB;
};

//! Class containing information about a super pixel
class SPInfo {
public:
  SPInfo() {neighbors=NULL; neighborsIDs=NULL; colorHist=NULL; colorProb=NULL; dispHist=NULL; dispProb=NULL;};
  ~SPInfo() {
    if(neighbors) 
      delete[] neighbors;
    if(neighborsIDs) 
      delete[] neighborsIDs;
    if(colorHist)
      delete colorHist;
    if(dispHist)
      delete dispHist;
    if(colorProb)
      delete[] colorProb;
    if(dispProb)
      delete[] dispProb;
  };
  void copy(SPInfo &spInfo) {
    label = 0;//segmInfo.label;
    neighbors = NULL;
    nrNeighbors = 0;
    //nrNeighbors = segmInfo.nrNeighbors;
    //neighbors = new int[nrNeighbors];
    //for(unsigned int i=0; i<nrNeighbors; i++)
    //  neighbors[i] = segmInfo.neighbors[i];
    nrPixels = spInfo.nrPixels;
    x = spInfo.x;
    y = spInfo.y;
    r = spInfo.r;
    g = spInfo.g;
    b = spInfo.b;
    //y = spInfo.y;
    //cr = spInfo.cr;
    //cb = spInfo.cb;
  };
  int label;
  int *neighbors;             //!< Integer array of size nrNeighbors containing the IDs of the neighboring super pixels
  int *neighborsIDs;          //!< Integer array of size nrSPixels containing the position of a neighbor in the 
                              //!< neighbors array for fast look up: neighborsIDs[neighbors[n]] == n
  unsigned int nrNeighbors;   //!< Number of neighbors of the super pixels
  int nrPixels;               //!< Number of conventional pixels in the super pixel
  float x, y;                 //!< Centroid of the super pixel
  float r,g,b;                //!< Mean r,g,b value of the super pixel.
  
  ColorHistogram *colorHist;  //!< Pointer to the color histogram
  DispHistogram *dispHist;    //!< Pointer to the disparity histogram. \b NB. Not used in super-pixel segmentation
  float *colorProb;           //!< Float array of size nrNeighbors containing the probabilities of this super pixel 
                              //!< being similar in color to its neighbors based on color histogram
  float *dispProb;            //!< Float array of size nrNeighbors containing the probabilities of this super pixel 
                              //!< being similar in disparity to its neighbors based on disparity histogram
  float dispMean, dispVar, dispStdError; //!<< mean, variance, and standard error of the disparity values contained in the super pixel
  bool validDisp;             //!< Boolean indicating if the disparity information of this super pixel is valid
  float cosTheta;             //!< Cos-angle between the normal of the super pixel and the normal of the dominant plane
  float distPlane;            //!< Distance of the super pixel to the dominant plane in the image
};

//! Struct containing the IDs of super pixels in a group. Used when super pixels are clustered
struct SPGroup {
  vector<int> group;
};

//! Struct containing top-down information of fore- or background for segmentation. 
struct TDInfo {
  ColorHistogram *colorHist;   //!< Color histogram 
  DispHistogram *dispHist;     //!< Disparity histogram. \b NB. Not used in segmentation
  float *colorProb;            //!< Float array of size nrSPixles with probabilities of being similar to all super-pixels based on color histograms
  float *dispProb;             //!< Float array of size nrSPixles with probabilities of being similar to all super-pixels based on disparity histograms
  float dispMean, dispVar;     //!< Mean and variance of disparities in the fore-/background
  float *dispProbFGauss;       //!< Float array of size nrSPixles with probabilities of being similar to all super-pixels based on disparity using Gaussian distribution
};

//! Class containing information for the graph-cut segmentation
class GraphCutInfo {
 public:
  GraphCutInfo() {
    fgInfo = new TDInfo();
    bgInfo = new TDInfo();
  };
  ~GraphCutInfo() {
    delete fgInfo;
    delete bgInfo;
  };
  SuperPixels *superPixels;  //!< Pointer to the super pixels
  TDInfo *fgInfo;            //!< Pointer to information about the foreground
  TDInfo *bgInfo;            //!< Pointer to information about the background
  float *nonPlaneProb;       //!< Float array of size nrSPixels containing the probabilities of all super pixels to \em not be part of the dominant plane
  int selectedFG;            //!< The selected foreground super pixel by the attention model
};

//! Struct containing the position of a point in interger values
struct Point {
  int x, y;
};

//! Struct containing the position of a point in float values
struct PointF{
  float x, y;
};


#endif
