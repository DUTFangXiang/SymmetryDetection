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

#ifndef MYTYPES2_H
#define MYTYPES2_H

#include <vector>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>

using namespace std;

//! Struct containing a vector of ints, which are the ids of pixels
struct Pixels {
  vector<int> pixels;
};

#endif
