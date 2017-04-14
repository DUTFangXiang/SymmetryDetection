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

#ifndef COLOR_H
#define COLOR_H

#include <math.h>

namespace Saliency {
  //! Class with arrays to do a conversion of a value to the jet color coding
  class Color {
  public:
    Color();
    ~Color();
    
    float *jetF32;         //!< Float array of size 3 * jetSize containing the RGB values of the jet colors (0.0 - 1.0)
    unsigned char *jetU8;  //!< Unsigned char array of size 3 * jetSize containing the RGB values of the jet colors (0 - 255)
    int jetSize;  
  };  
}
#endif
