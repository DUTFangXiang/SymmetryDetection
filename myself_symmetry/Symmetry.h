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

#ifndef SYMMETRY_H
#define SYMMETRY_H

//#define TIMING     //!< \def TIMING, Uncomment to display the timing
//#define DEBUG      //!< \def DEBUG, Uncomment to display some debugging info. \b NB. not well supported
//#define USE_OPENCL //!< \def USE_OPENCL, Uncomment to use OpenCL implementation for GPU calculations

#include <math.h>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <limits>
#include <vector>
#include <sys/utime.h>
#include <sys/stat.h>
#include <assert.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>

#include "unistd.h"
//FX:#include "Fixations.h"
#include "Color.h"
#include "myTypes.h"
#include "myTypes_sp.h"
//#include "tictoc.h"
//FX:#include <cl.h>
//#include <CL/cl.h>

using namespace std;

namespace Saliency {

#define SYMTYPE_ISOTROPIC  1
#define SYMTYPE_RADIAL     2

#define SYMSOURCE_BRIGHTNESSGRADIENT   1
#define SYMSOURCE_COLORGRADIENT        2
#define SYMSOURCE_BRIGHTNESS           3

#define SYMPERIOD_PI       1
#define SYMPERIOD_2PI      2

  //! Struct containing OpenCL information
  struct sCLEnv {

  //FX: cl_program program[1];
  //FX: cl_kernel kernel[7];
    
  //FX: cl_command_queue cmd_queue;
  //FX: cl_context context;
    
    size_t  buffer_size;
    size_t* global_worksize;
    size_t* local_worksize;
  };
  
  //! Struct containing the parameters for the symmetry-saliency method
  struct SymParams {
    int *scales;              //!< Int array of size nrScales containing the scales that need to be calculated relative to the first scale 
    int *userScales;          //!< Int array of size nrScales containing the absolute scales that need to be calculated (relative to input image)
    int nrScales;             //!< Number of scales
    int r1;                   //!< 'Radius' of the inner square of the symmetry kernel. Pixels within this square are not used for the symmetry calculation
    int r2;                   //!< 'Radius' of the outer square of the symmetry kernel. Pixels out side the inner and inside the outer square are used
    float s;                  //!< The sigma of the Gaussian weighting function on the distance between the pixels in a pixel pair
    int symmetryType;         //!< Type of symmetry calculation (SYMTYPE_ISOTROPIC or SYMTYPE_RADIAL)
    int symmetrySource;       //!< Information used by the symmetry calculation (SYMSOURCE_BRIGHTNESSGRADIENT, SYMSOURCE_COLORGRADIENT or SYMSOURCE_BRIGHTNESS). 
                              //!< SYMSOURCE_BRIGHTNESSGRADIENT works best, SYMSOURCE_COLORGRADIENT has not yet been implemented, and SYMSOURCE_BRIGHTNESS needs
                              //!< to be further developed.
    int symmetryPeriod;       //!< The period of the symmetry function (SYMPERIOD_PI or SYMPERIOD_2PI). The later is preferred.
    bool calcStrongestRadius; //!< Boolean to indicate whether the strongest contributing radii per pixel need to be calculated. Needed to get the proto-objects
    int seedR;                //!< The 'squared radius' of the area to find local maxima. Used only when proto-objects are calculated
    float seedT;              //!< Threshold on the seeds of the proto-objects (= local maxima in the saliency map)
    float roiT;               //!< Threshold on the region growing of proto-object base. Is relative to the seed.
    bool objThresholdAbsolute;//!< Boolean to indicate whether seedT is relative to the maximum saliency value, or an absolute threshold
  };

  //! Struct containing info of a symmetry pixel pair
  struct SymPair {
    float a0, a1;
    float symV;
  };

  //! Struct for a point in 2D
  struct Point2D {
    int x;
    int y;
  };

  //! Struct for an object point, position, radius, saliency value, and label
  struct ObjectPoint {
    int x;
    int y;
    float r;
    float v;
    int label;
  };

  //! Struct for a link between two labels
  struct LabelLink{
    int labelA;
    int labelB;
  };

  //! Class containing the map with the maximum contributing radii
  class RadMap {
  public:
    ~RadMap() { delete[] radiusMap; delete[] valueMap; };
    float *radiusMap;
    float *valueMap;
    int scale; // Relative to first scale 
    int userScale;
    int width;
    int height;
  };


  //! Class to calculate the symmetry-saliency of an image and select fixation points from the saliency map
  class Symmetry {
  public:
    Symmetry();
    ~Symmetry();
    void setParameters(SymParams symParams);
  
    void calcSaliencyMap(IplImage *img, int showScaleMaps = 1);
    Map *getSymmetryMap();
    IplImage *getSymmetryImage();
    vector<ObjectMap *>getObjectMap();
    int getScale();

    int debug;

  private:

    void calcSaliencyMapCPU(IplImage *img, int showScaleMaps = 1);
    void calcSaliencyMapCL(IplImage *img);

    // OpenCL
    void cl_init(sCLEnv& clEnv);
    char* cl_load_program_source(const char *filename);
    //void cl_gray(cl_mem rgbimg, cl_mem grayimg, sCLEnv& clEnv);
  
    void setImage(IplImage *_image);
    float getPatchSymmetry(float *angles, float *magnitudes, float *brightness, int x, int y);
    float getPatchSymmetry(float *angles, float *magnitudes, float *brightness, int x, int y, float &strongestRadius, float &strongestValue);
    float getPixelSymmetryGradient(float *angles, float *magnitudes, int x0, int y0, int x1, int x2, int d);
    float getPixelSymmetryBrightness(float *angles, float *magnitudes, float *brightness, int x0, int y0, int x1, int x2, int d);
    int normMap0to1(float* salMap);
    void normalizeMap(float* salMap);
    void addMapScale(float* sumMap, float* map);
    void multiplyMap(float *map, double val);
    IplImage *getMapImage(float *map);
    void showScaleMap(float *map, int width, int height, int sc);
    float cosFn(float x);
    float sinFn(float x);
    float logFn(float x);
    void clearRadiusMaps();
    bool localMaximum(float value, float *symMap, int x, int y, int width, int height);
    int toI(int x, int y, int width) { return y*width + x; };
    void sortOutLabels(vector<ObjectPoint> &seeds, int width, int height, vector<LabelLink> &labelLinks, int &nrLabels);
    void addRowLabels(bool **linkTable, uchar label, int nrLabels, vector<uchar> &group, bool *added);

    int width, height;
    int widthOriginal, heightOriginal;     //!< The original size of the image provided to calcSaliencyMap or calcSaliencyMapCL
    int widthSymMap, heightSymMap; //!< The size of the symmetry-saliency map. This is the first scale of the image pyramid used.
    int patchR1, patchR2;
    float sigma;
    float *distanceWeight;
    float **pixelAngles;
    int cosArSize;
    float *cosAr;
    int sinArSize;
    float *sinAr;
    int logArSize;
    float *logAr;
    int *scales;
    int *userScales;
    int nrScales;
    int symmetryType;
    int symmetrySource;
    int symmetryPeriod;
    bool calcStrongestRadius;
    int seedR;
    bool objThresholdAbsolute;
    float seedT;
    float roiT;

    IplImage *brightnessIpl;
    float *brightness;
    float *symMap ;

    Color *color;

    vector<RadMap*> strongestRadiusMaps;

    sCLEnv clEnv;
 
    IplImage *img;
  };

}

#endif
