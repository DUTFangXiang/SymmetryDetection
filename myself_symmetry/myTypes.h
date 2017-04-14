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

#ifndef MYTYPES_H
#define MYTYPES_H

#include <opencv/cv.h>
#include <opencv/highgui.h>

namespace Saliency {
  //! Class to store the saliency map and additional information
  class Map {
  public:
    Map() { map=NULL; };
    ~Map() { }; //if(map) delete[] map; }; // Not, because it is part of the Symmetry etc classes
    float *map;         //!< Pointer to the symmetry map
    int width;          //!< Width of the symmetry map
    int height;         //!< Height of the symmetry map
    int widthOriginal;  //!< Width of the original input image 
    int heightOriginal; //!< Height of the original input image 
    float sizeRatio;    //!< original size to symmetry-map size ratio
    float scale;        //!< Scale of the symmetry map
  };

  //! Class to store proto-object maps and additional information. \b NB. Currently not used for the super-pixel segmentation.
  class ObjectMap {
  public:
    ObjectMap() { map=NULL; descriptor=NULL;};
    ~ObjectMap() { if(map) delete[] map; if(descriptor) delete[] descriptor; };
    uchar *map;
    float *descriptor;
    int width, height;
    int x0, y0, x1, y1;
    float x, y;
    int x0sc, y0sc, x1sc, y1sc;
    float xsc, ysc; 
    float scale;
  };

  //! Struct to store position and saliency information if a fixation point
  struct Fixation {
    int x, y;
    float sal;
  };

  //! Struct to store the position and size of a bounding box
  struct SalRect {
    int l,r,t,b;
    float x,y;
  };

  //! Struct for the position of a pixel
  struct Pixel{ 
    float x; 
    float y; 
  };
 
  
  //! Struct containing information about the fixation point. 
  struct SaliencyPeak {
    float x;        //!< X-position in the image scale
    float y;        //!< Y-position in the image scale
    float xSc;      //!< X-position in the map scale 
    float ySc;      //!< Y-position in the map scale
    float sal;      //!< Peak saliency value
    float salNorm;  //!< Peak saliency value, normalized to maximum peak
  };

}

#endif
