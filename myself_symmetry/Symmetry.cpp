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

#include "Symmetry.h"
#include "additional.h"
using namespace Saliency;

/*!
 * Contructor. Creates look-up tables for the cosinus, sinus, and logarithm are created to 
 * speed up calculations. Furthermore sets some pointers to NULL and
 * creates an Color object for displayinf purposes.
 */
Saliency::Symmetry::Symmetry() {
  distanceWeight = NULL;
  pixelAngles = NULL;
  
  // Make cos-function from -4pi - 4pi
  cosArSize = 10000;
  cosAr = new float[2*cosArSize+1];
  for(int i=-cosArSize; i<cosArSize+1; i++) 
    cosAr[i+cosArSize] = cos(4*M_PI*(float)i/cosArSize);
  
  // Make sin-function from -4pi - 4pi
  sinArSize = 10000;
  sinAr = new float[2*sinArSize+1];
  for(int i=-sinArSize; i<sinArSize+1; i++) 
    sinAr[i+sinArSize] = sin(4*M_PI*(float)i/sinArSize);
  
  logArSize = 10000;
  logAr = new float[logArSize];
  for(int i=0; i<logArSize; i++) 
    logAr[i] = log( 1 + sqrt(72.0)*(float)i/(logArSize-1) ); // runs from 1 to 1+sqrt(72)
  
  scales = NULL;
  userScales = NULL;
  debug = 0;
  
  symMap = NULL;
  
  color = new Color;
  
  // Init OpenCL
#ifdef TIMING
  TIC;
#endif
  // Initialize OpenCL
#ifdef USE_OPENCL
  cl_init(clEnv);
#endif
#ifdef TIMING
  TOCus_("--init opencl--");
#endif
        
}

/*!
 * Destructor. Frees memory
 */
Saliency::Symmetry::~Symmetry() {
  delete[] distanceWeight;
  if(pixelAngles) {
    for(int i=0; i<patchR2*4+1; i++)
      delete[] pixelAngles[i];
    delete[] pixelAngles;
  }
  delete[] cosAr;
  delete[] sinAr;
  delete[] logAr;
  delete[] symMap;
  delete[] scales;
  delete[] userScales;
  clearRadiusMaps();
  delete color;
}

/*!
 * Sets the parameters for the symmetry-saliency calculations. Furthermore
 * look-up tables are created for the Gaussian distance weighting, and the 
 * angles between pixels in the symmetry kernel.
 * @param symParams a SymParams struct containing the parameters.
 */
void Saliency::Symmetry::setParameters(SymParams symParams) { //int r1, int r2, float s, int nrSc, int *sc) {

  std::cout << "Setting parameters\n";
  symmetryType = symParams.symmetryType;
  symmetryPeriod = symParams.symmetryPeriod;
  symmetrySource = symParams.symmetrySource;
  calcStrongestRadius = symParams.calcStrongestRadius;
  seedR = symParams.seedR;
  objThresholdAbsolute = symParams.objThresholdAbsolute;
  seedT = symParams.seedT;
  roiT = symParams.roiT;

  patchR1 = symParams.r1; // Pixels at this radius are included
  patchR2 = symParams.r2; // Pixels at this radius are included
  sigma = symParams.s;
  nrScales = symParams.nrScales;
  //if(scales)
  //	delete[] scales;
  //scales = new int[nrScales];
  //for(int i=0; i<nrScales; i++)
  //	scales[i] = symParams.scales[i];

  if(userScales)
    delete[] userScales;
  userScales = new int[nrScales];
  for(int i=0; i<nrScales; i++)
    userScales[i] = symParams.userScales[i];

  if(scales) 
    delete[] scales;
  scales = new int[nrScales];

  for(int s=0; s<nrScales; s++) {
    scales[s] = userScales[s] - userScales[0]; 
  }

  // Make a Gaussian distance weight array
  int lengthDW = 2 * ( (patchR2*2)*(patchR2*2) ) + 1;
  if(distanceWeight)
    delete[] distanceWeight;
  distanceWeight = new float[lengthDW];
  for(int i=0; i<lengthDW; i++) {
    distanceWeight[i] = (1/(sigma*sqrt(2*M_PI))) * exp( -i / (2*sigma*sigma));
  }

  if(pixelAngles) {
    for(int i=0; i<patchR2*4+1; i++)
      delete[] pixelAngles[i];
    delete[] pixelAngles;
  }

  pixelAngles = new float*[patchR2*4+1];
  for(int i=0; i<patchR2*4+1; i++)
    pixelAngles[i] = new float[patchR2*4+1];

  for(int y=-patchR2*2; y<patchR2*2+1; y++) {
    for(int x=-patchR2*2; x<patchR2*2+1; x++) {
      pixelAngles[y+patchR2*2][x+patchR2*2] = atan2((float)y, x);
    }
  }
}


/*!
 * Function to open OpenCL programs to do processing on a graphics card
 * @param filename is the name of the openCL file
 * @return Returns the source code of the file
 */
char* Saliency::Symmetry::cl_load_program_source(const char *filename) { 
	
  struct stat statbuf;
  FILE *fh; 
  char *source; 
	
  fh = fopen(filename, "r");
  if (fh == 0)
    return 0; 
	
  stat(filename, &statbuf);
  source = (char *) malloc(statbuf.st_size + 1);
  fread(source, statbuf.st_size, 1, fh);
  source[statbuf.st_size] = '\0'; 
	
  return source; 
} 

/*!
 * Initialization of OpenCL and loading of the kernels
 */
void Saliency::Symmetry::cl_init(sCLEnv& clEnv) {
//FX:暂且没用到
/***********************************************************
  cl_int err;

  cl_platform_id platform;
  cl_device_id device;
  
  // Get plattform
  err = clGetPlatformIDs(1, &platform, NULL);
  assert(err == CL_SUCCESS);

  // Find GPU device
  err = clGetDeviceIDs(NULL, CL_DEVICE_TYPE_GPU, 1,
                       &device, NULL);
  assert(err == CL_SUCCESS);

  // Create context
  clEnv.context = clCreateContext(0, 1, &device, NULL, NULL, &err);
  assert(err == CL_SUCCESS);

  // Create command queue
  clEnv.cmd_queue = clCreateCommandQueue(clEnv.context, device, 0, NULL);

  // Load program
  const char* filename = "symmetry.cl";
  char *program_source = cl_load_program_source(filename);
  clEnv.program[0] = clCreateProgramWithSource(clEnv.context, 1, (const char**)&program_source, NULL, &err);
  assert(err == CL_SUCCESS);

  // Build program
  err = clBuildProgram(clEnv.program[0], 0, NULL, NULL, NULL, NULL);

  cl_build_status build_status;
  clGetProgramBuildInfo(clEnv.program[0], device, CL_PROGRAM_BUILD_STATUS, sizeof(cl_build_status), &build_status, NULL);
  
  char *build_log;
  size_t ret_val_size = 2000;
  //clGetProgramBuildInfo(clEnv.program[0], device, CL_PROGRAM_BUILD_LOG, 0, NULL, &ret_val_size);
  build_log = new char[ret_val_size+1];
  clGetProgramBuildInfo(clEnv.program[0], device, CL_PROGRAM_BUILD_LOG, ret_val_size, build_log, NULL);
  //build_log[ret_val_size] = '\0';
  printf("Build Log:\n%s\n",build_log);

  //char build[2048];
  //clGetProgramBuildInfo(clEnv.program[0], device, CL_PROGRAM_BUILD_LOG, 2048, build, NULL);
#ifdef DEBUG
  printf("Build Log:\n%s\n",build);
#endif
  assert(err == CL_SUCCESS);
  
  // Load kernels
  clEnv.kernel[0] = clCreateKernel(clEnv.program[0], "gray", &err);
  assert(err == CL_SUCCESS);
  clEnv.kernel[1] = clCreateKernel(clEnv.program[0], "smooth", &err);
  assert(err == CL_SUCCESS);
  clEnv.kernel[2] = clCreateKernel(clEnv.program[0], "smoothresize", &err);
  assert(err == CL_SUCCESS);
  clEnv.kernel[3] = clCreateKernel(clEnv.program[0], "gradient", &err);
  assert(err == CL_SUCCESS);
  clEnv.kernel[4] = clCreateKernel(clEnv.program[0], "patch_symmetry", &err);
  assert(err == CL_SUCCESS);
  clEnv.kernel[5] = clCreateKernel(clEnv.program[0], "zero_buffer", &err);
  assert(err == CL_SUCCESS);
  clEnv.kernel[6] = clCreateKernel(clEnv.program[0], "add_symmetry_map", &err);
  assert(err == CL_SUCCESS);
  
  clEnv.global_worksize = NULL;
  clEnv.local_worksize  = NULL;
  **************************************************************************/
}


/*void Saliency::Symmetry::cl_gray(cl_mem rgbimg, cl_mem grayimg, sCLEnv& clEnv) {

  cl_int err;
  
  // Setup kernel arguments;
  err  = clSetKernelArg(clEnv.kernel[0],  0, sizeof(cl_mem), &rgbimg);
  assert(err == CL_SUCCESS);
  err |= clSetKernelArg(clEnv.kernel[0],  1, sizeof(cl_mem), &grayimg);
  assert(err == CL_SUCCESS);

  // Enqueue the kernel
  err = clEnqueueNDRangeKernel(clEnv.cmd_queue, clEnv.kernel[0], 1, NULL, 
  &clEnv.worksize, NULL, 0, NULL, NULL);
  assert(err == CL_SUCCESS);
  
  
  }
*/


/*!
 * Calculation of the symmetry-saliency map. A parallel implementation that 
 * can run on the graphics card using OpenCL. The symmetry
 * values are calculated for all pixels in the image, on different scales. The different scales
 * are resized to the first scale and the values are added together to get the symmetry-
 * saliency map.
 * @param _img the camera image
 */
void Saliency::Symmetry::calcSaliencyMapCL(IplImage* _img) {
  img = _img;
}

void Saliency::Symmetry::calcSaliencyMap(IplImage *img, int showScaleMaps) {
#ifdef USE_OPENCL
  calcSaliencyMapCL(img, showScaleMaps);
#else
  calcSaliencyMapCPU(img, showScaleMaps);
#endif
}

/*!
 * Calculation of the symmetry-saliency map. The standard serial cpu implementation. The symmetry
 * values are calculated for all pixels in the image, on different scales. The different scales
 * are resized to the first scale and the values are added together to get the symmetry-
 * saliency map. The saliency map is internally stored and can be retrieved by subsequently calling 
 * getSymmetryMap or getSymmetryImage.
 * If calcStrongestRadius==true, the strongest contributing radius per pixel and per scale is
 * calculated, so that the symmetry area can be determined.
 * 
 * @param _img the camera image. IplImage pointer with IPL_DEPTH_8U and 3 color channels (BGR)
 * @param showScaleMaps integer to control the display of the scale maps. (0 is off 1 is on)
 * @see See calcSaliencyMapCL for a parallel implementation on the graphics card.
 * @see Use getSymmetryMap to get the symmetry-saliency map.
 * @see Use getSymmetryImage to get the symmetry-saliency map as an image.
 * @see See getPatchSymmetry(angles, magnitudes, (float *)imgCurrScale->imageData, x, y) for
 *      the calculation of the symmetry value for a given pixel
 * @see See getPatchSymmetry(angles, magnitudes, (float *)imgCurrScale->imageData, x, y, strongestRadius, strongestValue)
 *      for the calculation of the symmetry value for a given pixel and the calculation of
 *      the strongest contributing radius
 * @see See addMapScale(symMap, sMapScale) for the cross-scale addition of symmetry scale maps
 */
void Saliency::Symmetry::calcSaliencyMapCPU(IplImage *_img, int showScaleMaps) {
  img = _img;
  //转化图片到图片金字塔的第一规模图，
  //并且得到第一规模图灰度图brightnessIpl
  setImage(img);
  // Create a clean symMap
  //FX：此处先注释掉
   if(symMap)
    delete[] symMap;
   symMap = new float[widthSymMap*heightSymMap];
  // change to memset? faster?
  for(int i=0; i<widthSymMap*heightSymMap; i++)
    symMap[i] = 0;

  // // Make an IplImage of brightness
  //IplImage *brightnessImg = cvCreateImageHeader(cvSize(widthSymMap, heightSymMap), IPL_DEPTH_32F, 1);
  //cvSetData(brightnessImg, brightness, widthSymMap*sizeof(float));

  IplImage *imgPrevScale = cvCloneImage(brightnessIpl);
  IplImage *imgCurrScale;
	
  // Calculate symmetry map per scale
  int lastScale = 0;

  if(calcStrongestRadius)
    clearRadiusMaps();

  for(int sc=0; sc<nrScales; sc++) {
    if(debug) printf("    radSym: scale %d of %d\n", sc, nrScales);
    // Resize the image
    if(scales[sc]!=0) {
      if(debug) printf("    radSym: scale down image\n");
      width = (int)rint((float)widthSymMap / pow(2.0, (double)scales[sc])); // << instead of pow
	  //<math.h>包含rint
      height = (int)rint((float)heightSymMap / pow(2.0, (double)scales[sc]));
      imgCurrScale = cvCreateImage(cvSize(width, height), IPL_DEPTH_32F, 1);
      cvSmooth(imgPrevScale, imgPrevScale, CV_GAUSSIAN, 3);
      cvResize(imgPrevScale, imgCurrScale);
      cvReleaseImage(&imgPrevScale);
      lastScale = scales[sc];
    } else {
      if(debug) printf("    radSym: set scale image\n");
      imgCurrScale = imgPrevScale;
      width = widthSymMap;
      height = heightSymMap;
    }

    // create scale map and a strongest radius map
    float *sMapScale = new float[width*height];
    float *strongestRadiusMapScale=NULL;
    float *strongestValueMapScale=NULL;
    if(calcStrongestRadius) {
      strongestRadiusMapScale = new float[width*height];
      strongestValueMapScale = new float[width*height];
    }

    // Get the gradients in the image using opencv functions
	//获得图像梯度
    if(debug) printf("    radSym: calc gradients\n");
    IplImage *imgX = cvCreateImage(cvGetSize(imgCurrScale), IPL_DEPTH_32F, 1);
    IplImage *imgY = cvCreateImage(cvGetSize(imgCurrScale), IPL_DEPTH_32F, 1);
	//通常情况，cvSobel函数调用采用如下参数 (xorder=1, yorder=0, aperture_size=3) 
	//或 (xorder=0, yorder=1, aperture_size=3) 来计算一阶 x- 或 y- 方向的图像差分
	//imgCurrScale是输入图, imgX输出图
    cvSobel(imgCurrScale, imgX, 1, 0, 3);
    cvSobel(imgCurrScale, imgY, 0, 1, 3);
		
    // Calculate the angles and magnitudes
	//获取图像梯度的角度和大小
    if(debug) printf("    radSym: calc angles / magn\n");
    float *_imgX = (float*)imgX->imageData;
    float *_imgY = (float*)imgY->imageData;
    float *angles = new float[width*height];
    float *magnitudes = new float[width*height];

    for(int i=0; i<width*height; i++) {
      angles[i] = atan2(_imgY[i], _imgX[i]);
      magnitudes[i] = sqrt(_imgY[i]*_imgY[i] + _imgX[i]*_imgX[i]);
    }

    //获取对称map
    int i;
    float strongestRadius, strongestValue;
    if(debug) printf("    radSym: getPatchSymmetry\n");
    for(int y=0; y<height; y++) {
      for(int x=0; x<width; x++) {
	//对每个像素进行排号，一一计算其对称值
	i = y*width + x;
	//calcStrongestRadius =1，每个像素和每个规模的最强contributing radius计算了，所以对称区域就可以被确定
	if(calcStrongestRadius) {
	  //计算每个像素的对称值的函数
	  sMapScale[i] = getPatchSymmetry(angles, magnitudes, (float *)imgCurrScale->imageData, x, y, strongestRadius, strongestValue);
	  strongestRadiusMapScale[i] = strongestRadius;
	  strongestValueMapScale[i] = strongestValue;
	}
	else
	  sMapScale[i] = getPatchSymmetry(angles, magnitudes, (float *)imgCurrScale->imageData, x, y);
      }
    }
	showScaleMaps = 0;  //不显示图片为0
    if(showScaleMaps)
      showScaleMap(sMapScale, width, height, sc);

    if(debug) printf("    radSym: normalize and add\n");
    //normalizeMap(sMapScale); // TODO: experiment with this function
    addMapScale(symMap, sMapScale);

    imgPrevScale = imgCurrScale;

    // Add the radius map to the vector radiusMaps
    RadMap *strongestRadiusMap;
    if(calcStrongestRadius) {
      strongestRadiusMap = new RadMap;
      strongestRadiusMap->radiusMap = strongestRadiusMapScale;
      strongestRadiusMap->valueMap = strongestValueMapScale;
      strongestRadiusMap->userScale = userScales[sc]; // Relative to first scale of IK model
      strongestRadiusMap->scale = scales[sc]; // Relative to first scale of IK model
      strongestRadiusMap->width = width;
      strongestRadiusMap->height = height;
      strongestRadiusMaps.push_back(strongestRadiusMap);
    }

    cvReleaseImage(&imgX);
    cvReleaseImage(&imgY);
    delete[] angles;
    delete[] magnitudes;
    delete[] sMapScale;
  }

  if(showScaleMaps) {
    cvDestroyWindow("scale0");
    cvDestroyWindow("scale1");
    cvDestroyWindow("scale2");
  }

  // Normalize
  normMap0to1(symMap);


  cvReleaseImage(&imgCurrScale);
  cvReleaseImage(&brightnessIpl);
}

/*!
 * Set the camera image, transforms the uchar values to floats (range 0-1), resizes the
 * image to the first scale used by the symmetry method by iteratively applying Gaussian 
 * bluring and down sampling and converts it to gray scale.
 * 
 * @param _image the camera image
 */
void Saliency::Symmetry::setImage(IplImage *_image) {

  // Set the original size of the input image
  widthOriginal = _image->width;
  heightOriginal = _image->height;

  // Step 1: from 8U to 32F
  IplImage *image = cvCreateImage(cvGetSize(_image),IPL_DEPTH_32F, 3 );
  //transforms the uchar values to floats (range 0-1)
  cvConvertScale(_image, image, (double)1.0/256);
  width = _image->width;
  height = _image->height;
	
  // Step 2: resize image to first defined scale
  //高斯滤波3次，转化图片到 first scale
  int firstScale = userScales[0];
  // Iteratively smooth and down scale image
  for(int s=0; s<firstScale; s++) {
    width = (int)(width/2);
    height = (int)(height/2);
    cvSmooth(image, image, CV_GAUSSIAN, 3);
    IplImage *tempImg = cvCreateImage(cvSize(width, height),IPL_DEPTH_32F, 3 );
	//重新调整图像image，使它精确匹配目标tempImg
    cvResize(image, tempImg);
    cvReleaseImage(&image);
    image = cvCloneImage(tempImg);
    cvReleaseImage(&tempImg);
  }

  widthSymMap = width;
  heightSymMap = height;

  // Step 3: get brightness image
  brightnessIpl = cvCreateImage( cvGetSize(image),IPL_DEPTH_32F,1 );
  //参数CV_RGB2GRAY是RGB到gray转换
  cvCvtColor(image, brightnessIpl, CV_RGB2GRAY);
  //灰度图输出结果是brightnessIpl
  brightness = (float *)brightnessIpl->imageData;

  cvReleaseImage(&image);
}


/*!
 * Provides the symmetry map and some additional information
 * @return Returns a pointer to a a Map object, containing a pointer to the symmetry map, 
 * the width and hight of the symmetry map, the width and height of the original input
 * image, the the original to symmetry-map size ratio, and the scale of the symmetry map.
 * Scale 0 is the original size of the image, scale 1 is the image down sized
 * by a factor two, scale 2, is down sized twice, etc. cvReleaseImage need to be called by the user.
 */
Map *Saliency::Symmetry::getSymmetryMap() {
  // Should not be deleted by user
  Map *map = new Map;
  map->map = symMap;
  map->width = widthSymMap;
  map->height = heightSymMap;
  map->widthOriginal = widthOriginal;
  map->heightOriginal = heightOriginal;
  map->sizeRatio = (float)widthOriginal/(float)widthSymMap;
  map->scale = (float)userScales[0];
  return map;
}

/*!
 * Provides an image containing the symmetry map in JET colors.
 * @return Returns a pointer to a IplImage containing the symmetry map,
 * where the symmetry values are coded as JET colors. 
 */
IplImage *Saliency::Symmetry::getSymmetryImage() {
  IplImage *jetImage = getMapImage(symMap);
  return jetImage;
}

/*!
 * Gets the symmetry value for the pixel (x,y). The symmetry contributions for all pixel pairs in the 
 * kernel (patch) are summed weighted by the distance between the pixels. The symmetry kernel is 
 * defined with patchR1 and patchR2, so that pixels in a squared-donut shape are considered.
 * If symmetrySource==SYMSOURCE_BRIGHTNESSGRADIENT, the symmetry is calculated based on the
 * brightness gradients. If the private member variable symmetrySource==SYMSOURCE_BRIGHTNESS, the 
 * symmetry is calculated based on the gradients and the brightness values. The later needs to be 
 * further tested and developed. If the private member symmetryType==SYMTYPE_RADIAL, the maximum
 * contributing pixel pair is calculated and used to promote patterns with multiple axes of
 * symmetry. Otherwise (SYMTYPE_ISOTROPIC), the contributions of the pixel pairs are summed.
 *
 * @param angles float array containing the gradient angles (rad) of all pixels 
 * @param magnitudes float array containing the gradient magnitudes of all pixels 
 * @param brightness float array containing the brightness values of all pixels
 * @param x the x coordinate of the center pixel
 * @param y the y coordinate of the center pixel
 * @return the symmetry value for pixel (x,y)
 * @see See getPixelSymmetryGradient for the calculation of the symmetry distribution of one pixel
 *      pair using gradient information 
 * @see See getPixelSymmetryBrightness for the calculation of the symmetry distribution of one pixel
 *      pair using gradient and brightness information
 * @see Use getPatchSymmetry(float *angles, float *magnitudes, float *brightness, int x, int y, float &strongestRadius, float &strongestValue)
 *      if the strongest contributing radius per pixel needs to be calculated in order to determine
 *      the symmetry area.
 */
float Saliency::Symmetry::getPatchSymmetry(float *angles, float *magnitudes, float *brightness, int x, int y) {
  // Gets the symmetry for pixel x,y. Sets the strongest contributing radius in strongestRadius
  //获得像素(x,y)的对称值. 对称贡献是多有像素对在kernel (patch)里面
  // kernel (patch)的和被像素的距离衡量. The symmetry kernel is 
  // defined with patchR1 and patchR2, 所以像素在方形环状里面才被考虑。
  float totalSym = 0;
  int x0, y0, x1, y1, d; //, i0, i1;

  // If the mask is close to the borders, process only the part that is within the valid angle and magnitude values.
  // Excluding the borders, since the gradients there are not vaild
  //不算边界点像素的对称值
  int dy1 = max(patchR2 - y +1, 1);
  int dy2 = max(y + patchR2 + 1 - height + 1, 1);
  int dy = max(dy1, dy2);

  int dx1 = max(patchR2 - x +1 ,1);
  int dx2 = max(x + patchR2 + 1 - width + 1, 1);  
  int dx = max(dx1, dx2);

  SymPair maxSymPair; maxSymPair.a0 = 0; maxSymPair.a1 = 0;
  maxSymPair.symV = 0;
  SymPair *symPairs = NULL;
  if(symmetryType==SYMTYPE_RADIAL)
    symPairs = new SymPair[(2*patchR2+1)*(2*patchR2+1)]; // Maximum size is /2
  int nrPairs = 0;
	
  float dX, dY;
  float symV = 0;

  for(int j=dy; j < patchR2+1; j++) {
    for(int i = dx; i < patchR2*2+1 - dx; i++) {
      if(j >= patchR2 && i >= patchR2)  // When at the center of the mask, break
	break;
      if( !(j>(patchR2-patchR1) && j<(patchR2+patchR1) && i>(patchR2-patchR1) && i<(patchR2+patchR1)) ) {
	x0 = x - patchR2 + i;
	y0 = y - patchR2 + j;
	x1 = x + patchR2 - i;
	y1 = y + patchR2 - j;
	dX = x1 - x0;
	dY = y1 - y0;
              
	d = (int)rint(dX*dX+dY*dY);  // L1 distance
	if(symmetrySource == SYMSOURCE_BRIGHTNESSGRADIENT)
	  symV = getPixelSymmetryGradient(angles, magnitudes, x0, y0, x1, y1, d);
	if(symmetrySource == SYMSOURCE_BRIGHTNESS)
	  symV = getPixelSymmetryBrightness(angles, magnitudes, brightness, x0, y0, x1, y1, d);
              
	if(symmetryType==SYMTYPE_ISOTROPIC)
	  totalSym += symV;  
	else {
	  symPairs[nrPairs].a0 = angles[y0*width + x0];
	  symPairs[nrPairs].a1 = angles[y1*width + x1];
	  symPairs[nrPairs].symV = symV;
	  if(symV > maxSymPair.symV) {
	    maxSymPair = symPairs[nrPairs];
	  }
	  nrPairs++;
	}
      }
    }
  }

  if(symmetryType==SYMTYPE_ISOTROPIC)
    return totalSym;
  else if(symmetryType==SYMTYPE_RADIAL) {
    if(maxSymPair.symV == 0)
      return totalSym;
		
    float domSymDir = (maxSymPair.a0 + maxSymPair.a1) / 2;
    float symDir;
    for(int i=0; i<nrPairs; i++) {
      symDir = (symPairs[i].a0 + symPairs[i].a1) / 2;
      totalSym += symPairs[i].symV * sin(symDir-domSymDir)*sin(symDir-domSymDir);
    }
				
    delete symPairs;
  }

  return totalSym;	
}
 
/*!
 * Gets the symmetry value for the pixel (x,y) and calculates the strongest contributing radius, 
 * so that the symmetry area can be calculated. The symmetry contributions for all pixel pairs in the 
 * kernel (patch) are summed weighted by the distance between the pixels. The symmetry kernel is 
 * defined with patchR1 and patchR2, so that pixels in a squared-donut shape are considered.
 * If symmetrySource==SYMSOURCE_BRIGHTNESSGRADIENT, the symmetry is calculated based on the
 * brightness gradients. If the private member variable symmetrySource==SYMSOURCE_BRIGHTNESS, the 
 * symmetry is calculated based on the gradients and the brightness values. The later needs to be 
 * further tested and developed. If the private member symmetryType==SYMTYPE_RADIAL, the maximum
 * contributing pixel pair is calculated and used to promote patterns with multiple axes of
 * symmetry. Otherwise (SYMTYPE_ISOTROPIC), the contributions of the pixel pairs are summed.
 *
 * @param angles float array containing the gradient angles (rad) of all pixels 
 * @param magnitudes float array containing the gradient magnitudes of all pixels 
 * @param brightness float array containing the brightness values of all pixels
 * @param x the x coordinate of the center pixel
 * @param y the y coordinate of the center pixel
 * @param strongestRadius float array to store the strongest contributing radii
 * @param strongestValue float array to store the strongest contributing values
 * @return the symmetry value for pixel (x,y)
 * @see See getPixelSymmetryGradient for the calculation of the symmetry distribution of one pixel
 *      pair using gradient information 
 * @see See getPixelSymmetryBrightness for the calculation of the symmetry distribution of one pixel
 *      pair using gradient and brightness information
 * @see Use getPatchSymmetry(float *angles, float *magnitudes, float *brightness, int x, int y)
 *      if the strongest contributing radii do not need to be calculated
 */
float Saliency::Symmetry::getPatchSymmetry(float *angles, float *magnitudes, float *brightness, int x, int y, float &strongestRadius, float &strongestValue) {
  // Gets the symmetry for pixel x,y. Sets the strongest contributing radius in strongestRadius
  //获得像素(x,y)的对称值. 对称贡献是多有像素对在kernel (patch)里面
  // kernel (patch)的和被像素的距离衡量. The symmetry kernel is 
  // defined with patchR1 and patchR2, 所以像素在方形环状里面才被考虑。
  //不算边界点像素的对称值	 
  float totalSym = 0;
  int x0, y0, x1, y1, d; //, i0, i1;
  // patchR1 = symParams.r1 = 3；
  // patchR2 = symParams.r2 = 8; //Pixels at this radius are included
  // Excluding the borders, since the gradients there are not vaild
  int dy1 = max(patchR2 - y +1, 1);
  int dy2 = max(y + patchR2 + 1 - height + 1, 1);
  int dy = max(dy1, dy2);

  int dx1 = max(patchR2 - x +1 ,1);
  int dx2 = max(x + patchR2 + 1 - width + 1, 1);  
  int dx = max(dx1, dx2);


  SymPair maxSymPair; maxSymPair.a0 = 0; maxSymPair.a1 = 0;
  maxSymPair.symV = 0;
  SymPair *symPairs = NULL;
  if(symmetryType==SYMTYPE_RADIAL)
    symPairs = new SymPair[(2*patchR2+1)*(2*patchR2+1)]; // Maximum size is /2
  int nrPairs = 0;
	
  float dX, dY;
  float symV = 0;

  float strongestSymV = 0;
  float strongestSymR = 0;
        
  for(int j=dy; j < patchR2+1; j++) {
    for(int i = dx; i < patchR2*2+1 - dx; i++) {
      if(j >= patchR2 && i >= patchR2) {  // When at the center of the mask, break
	break;
      }
      if( !(j>(patchR2-patchR1) &&
	    j<(patchR2+patchR1) &&
	    i>(patchR2-patchR1) &&
	    i<(patchR2+patchR1)) ) {
              
	x0 = x - patchR2 + i;
	y0 = y - patchR2 + j;
	x1 = x + patchR2 - i;
	y1 = y + patchR2 - j;
	dX = x1 - x0;
	dY = y1 - y0;
             
	d = (int)rint(dX*dX+dY*dY);
	//定义中symmetrySource = SYMSOURCE_BRIGHTNESSGRADIENT
	if(symmetrySource == SYMSOURCE_BRIGHTNESSGRADIENT)
	  //重要处理过程
	  symV = getPixelSymmetryGradient(angles, magnitudes, x0, y0, x1, y1, d);
	if(symmetrySource == SYMSOURCE_BRIGHTNESS)
	  symV = getPixelSymmetryBrightness(angles, magnitudes, brightness, x0, y0, x1, y1, d);

    //定义中SymmetryType = SYMTYPE_ISOTROPIC;          
	if(symmetryType==SYMTYPE_ISOTROPIC)
	  totalSym += symV;  
	else {
	  symPairs[nrPairs].a0 = angles[y0*width + x0];
	  symPairs[nrPairs].a1 = angles[y1*width + x1];
	  symPairs[nrPairs].symV = symV;
	  if(symV > maxSymPair.symV) {
	    maxSymPair = symPairs[nrPairs];
	  }
	  nrPairs++;
	}
	// Find radius maximally contribution to the total symmetry
	if(symV > strongestSymV) {
	  strongestSymV = symV;
	  strongestSymR = d;  // For computational reasons, the square of the diameter is given instead of the radius (r = sqrt(d)/2)
	}
      }
    }
  }
        
  strongestRadius = strongestSymR;
  strongestValue = strongestSymV;

  if(symmetryType==SYMTYPE_ISOTROPIC)
    return totalSym;
  else if(symmetryType==SYMTYPE_RADIAL) {
    if(maxSymPair.symV > 0) {
      float domSymDir = (maxSymPair.a0 + maxSymPair.a1) / 2;
      float symDir;
      for(int i=0; i<nrPairs; i++) {
	symDir = (symPairs[i].a0 + symPairs[i].a1) / 2;
	totalSym += symPairs[i].symV * sin(symDir-domSymDir)*sin(symDir-domSymDir);
      }
    }				
    delete symPairs;
    return totalSym;
  }

  return totalSym;	
}

/*!
 * Calculates the symmetry contribution of a single pixel pair in the symmetry kernel using
 * brightness gradients. This function is based on the symmetry operator of Reisfeld \em et \em al. (1995)
 * If the member variable symmetryPeriod==SYMPERIOD_PI, a pi-periodic function is used. This is taken from 
 * Heidemann (2004). This method works less well in my experience. If 
 * symmetryPeriod==SYMPERIOD_2PI, the 2pi-periodic function of Reisfeld \em et \em al is used
 *
 * @param angles float array containing the gradient angles (rad) of all pixels 
 * @param magnitudes float array containing the gradient magnitudes of all pixels 
 * @param x0 the x coordinate of the first pixel
 * @param y0 the y coordinate of the first pixel
 * @param x1 the x coordinate of the second pixel
 * @param y1 the y coordinate of the second pixel
 * @param d the distance between the two pixels
 * @return a float with the symmetry contribution
 */
inline float Saliency::Symmetry::getPixelSymmetryGradient(float *angles, float *magnitudes, int x0, int y0, int x1, int y1, int d) {
  //使用亮度梯度，计算在对称核中单个像素对的对称贡献量
  // Get the pixel indices
  int i0 = y0 * width + x0;
  int i1 = y1 * width + x1;

  // Get the angle of the line between the two pixels
  float angle = pixelAngles[(y1-y0)+patchR2*2][(x1-x0)+patchR2*2];

  // Subtract the angle between the two pixels from the gradient angles to get the normalized angles
  float g0 = angles[i0] -  angle;
  float g1 = angles[i1] -  angle;
	
  // Calculate the strength of both gradient magnitudes
  float gwf = logFn( 1+magnitudes[i0] ) * logFn( 1+magnitudes[i1] ); // logFn uses the look-up table
	
  // Return the symmetry contribution by comparing the normalized gradient angles multiplied by the
  // gradient strength and the distance weight. Different formula are used depending on the value of
  // symmetryPeriod.
  if(symmetryPeriod == SYMPERIOD_PI)
    return( ( cosFn(g0+g1)*cosFn(g0+g1) * cosFn(g0)*cosFn(g0) * cosFn(g1)*cosFn(g1) ) * gwf * distanceWeight[d]  );
  else if(symmetryPeriod == SYMPERIOD_2PI)
    return(  (1 - cosFn(g0 + g1))*(1 - cosFn(g0 - g1)) * gwf * distanceWeight[d]  );
  else
    return(0);
}

/*!
 * \b NB. Under construction and functionality not fully tested. \n
 * Calculates the symmetry contribution of a single pixel pair in the symmetry kernel using
 * brightness gradients and brightness values. The function is based on the symmetry operator of
 * Reisfeld \em et \em al. (1995) and extended by using the difference in brightness between the
 * two pixels. Furthermore the function is identical to 
 * getPixelSymmetryGradient(float *angles, float *magnitudes, int x0, int y0, int x1, int y1, int d)
 * If the member variable symmetryPeriod==SYMPERIOD_PI, a pi-periodic function is used. This is taken from 
 * Heidemann (2004). This method works less well in my experience. If 
 * symmetryPeriod==SYMPERIOD_2PI, the 2pi-periodic function of Reisfeld \em et \em al is used
 * 
 * @param angles float array containing the gradient angles (rad) of all pixels 
 * @param magnitudes float array containing the gradient magnitudes of all pixels 
 * @param brightness float array containing the brightness values of all pixels
 * @param x0 the x coordinate of the first pixel
 * @param y0 the y coordinate of the first pixel
 * @param x1 the x coordinate of the second pixel
 * @param y1 the y coordinate of the second pixel
 * @param d the distance between the two pixels
 * @return a float with the symmetry contribution
 * @see See getPixelSymmetryGradient(float *angles, float *magnitudes, int x0, int y0, int x1, int y1, int d)
 *      for the calculation using the brightness gradients only
 */
inline float Saliency::Symmetry::getPixelSymmetryBrightness(float *angles, float *magnitudes, float *brightness, int x0, int y0, int x1, int y1, int d) {
  // Make this function inline
  int i0 = y0 * width + x0;
  int i1 = y1 * width + x1;

  //float angle = atan2( (y1-y0), (x1-x0) );
  //float angle = pixelAngles[(y1-y0)][(x1-x0)];
  float angle = pixelAngles[(y1-y0)+patchR2*2][(x1-x0)+patchR2*2];

  // eqn 5
  float g0 = angles[i0] -  angle;
  float g1 = angles[i1] -  angle;
	
  // eqn 6
  float gwf = logFn( 1+magnitudes[i0] ) * logFn( 1+magnitudes[i1] ); // Replaced to increase speed
  //float gwf = log( 1+magnitudes[i0] ) * log( 1+magnitudes[i1] ); // Replaced to increase speed
  //float gwf = magnitudes[i0]*magnitudes[i1];
	
  // eqn 3
  if(symmetryPeriod == SYMPERIOD_PI)
    return( (1 - fabs(brightness[i0]-brightness[i1])) * ( cosFn(g0+g1)*cosFn(g0+g1) * cosFn(g0)*cosFn(g0) * cosFn(g1)*cosFn(g1) )  *  gwf * distanceWeight[d] );
  else //if(symmetryPeriod == SYMPERIOD_2PI)
    //这个效果更好，所以使用该方式
    return( (1 - fabs(brightness[i0]-brightness[i1])) * (1 - cosFn(g0 + g1))*(1 - cosFn(g0 - g1))  *  gwf * distanceWeight[d] );

}

/*!
 * Scales the values in the salMap between 0 and 1
 * @param salMap a float array containing the not-normalized saliency map
 * @return Returns an int. 0 in case the salMap contains all the same values.
 * In which case all values in salMap are set to 0.0. 1 is returned otherwise.
 */
int Saliency::Symmetry::normMap0to1(float* salMap) {

  float min = numeric_limits<double>::max();
  float max = numeric_limits<double>::min();

  int w = widthSymMap;
  int h = heightSymMap;

  for(int i=0; i<w*h; i++) {
    if(salMap[i] < min)
      min = salMap[i];
    if(salMap[i] > max)
      max = salMap[i];
  }

  if(max>min) {
    for(int i=0; i<w*h; i++)
      salMap[i] = (salMap[i]-min) / (max-min);
    return(1);
  } else {
    for(int i=0; i<w*h; i++)
      salMap[i] = 0.0;
    return(0);
  }
}

/*!
 * This function implements the feature-map normalization method, N(.), used by 
 * Itti \em et \em al (1998, page 2). The method first normalized the map to 
 * values between 0 and 1, then finds all local maxima in a rectangle of size 
 * 2*maxR+1. The map is then multiplied by the squared difference between
 * the average value of the local maxima and the global maximum (1.0 due to 
 * initial normalization).
 * @param salMap a float array containing the not-normalized saliency map. The
 * values in this array will be altered. 
 */
void Saliency::Symmetry::normalizeMap(float* salMap) {
  int w = widthSymMap;
  int h = heightSymMap;

  int res = normMap0to1(salMap);
  if(res) {
    int maxR = 3;
		
    // Calculate the average value of all local minima
    float sumMax = 0.0, maxVal;
    int nrMax = 0;
    float pixVal, val;
    for(int y=maxR; y<h-maxR; y++) {
      for(int x=maxR; x<w-maxR; x++) {
	pixVal = salMap[y*w+x];
	maxVal = -1;
	for(int my=-maxR; my<=maxR; my++) {
	  for(int mx=-maxR; mx<=maxR; mx++) {
	    val = salMap[(y+my)*w + (x+mx)];
	    if(val>maxVal)
	      maxVal = val;
	  }
	}
	if(maxVal==pixVal) {
	  nrMax++;
	  sumMax += pixVal;
	}
      }
    }
    //nrMax--;
    sumMax -= 1; // Subtract the global maximum

    if(nrMax>0) {
      float maxAv = sumMax/nrMax;
      // Multiply the map ;
      multiplyMap(salMap, (1-maxAv)*(1-maxAv));
    }
  }
}

/*
 * Adds the values in scaleMap to the values in sumMap.
 * @param sumMap float array to which scaleMap will be added
 * @param scaleMap float array with the values that will be added
 */
void Saliency::Symmetry::addMapScale(float* sumMap, float* scaleMap) {
  // Resize map to sumMap size using IplImage
  IplImage *scaleI = cvCreateImageHeader(cvSize(width, height), IPL_DEPTH_32F, 1);
  cvSetData(scaleI, scaleMap, width*sizeof(float));
  
  IplImage *mapI = cvCreateImage(cvSize(widthSymMap, heightSymMap), IPL_DEPTH_32F, 1);
  cvResize(scaleI, mapI);
  
  float *map = (float *)mapI->imageData;
  
  for(int i=0 ;i< widthSymMap*heightSymMap; i++) {
    sumMap[i] += map[i];
  }
  
  cvReleaseImageHeader(&scaleI);
  cvReleaseImage(&mapI);
}


/*
 * Multiplies the map with val
 * @param map float array. The values in the array will be multiplied with val
 * @param val the value with which the map is multiplied
 */
void Saliency::Symmetry::multiplyMap(float *map, double val) {
  for(int i=0; i<width*height; i++) 
    map[i] = val * map[i];
}

/*
 * Takes the values in map and makes a image displaying the values using
 * the JET color scheme. The map is normalized to the size of the original
 * input image. The values in map are normalized between 0 and 1.
 * \b NB. the normalization will alter the values in map.
 * @param map float array
 * @return Returns a pointer to an IplImage, IPL_DEPTH_*U, 3 color channels. 
 *         cvReleaseImage need to be called by the user.
 */
IplImage *Saliency::Symmetry::getMapImage(float *map) {
  // Normalize map
  normMap0to1(map);

  // Convert maps to jet images
  IplImage *jetImage = cvCreateImage(cvSize(widthSymMap, heightSymMap), IPL_DEPTH_8U, 3);
  char *jetData = jetImage->imageData;
  int jetI,i,ii;
  for(int y=0; y<heightSymMap; y++) {
    for(int x=0; x<widthSymMap; x++) {
      i = y*jetImage->widthStep + jetImage->nChannels*x;
      ii = y*widthSymMap + x;
      jetI = (int)floor(map[ii]*(color->jetSize-1));
      jetData[i] = (char)rint(color->jetU8[3*jetI+2]);
      jetData[i+1] = (char)rint(color->jetU8[3*jetI+1]);
      jetData[i+2] = (char)rint(color->jetU8[3*jetI]);
    }
  }

  // Resize to origninal size
  IplImage *iplImage;
  if(userScales[0]>0) {
    //int dstW = (int)(widthSymMap * pow(2.0,(double)userScales[0]));
    //int dstH = (int)(heightSymMap * pow(2.0,(double)userScales[0]));
    iplImage = cvCreateImage(cvSize(widthOriginal, heightOriginal), IPL_DEPTH_8U, 3);
    cvResize(jetImage, iplImage);
    cvReleaseImage(&jetImage);
  } else {
    iplImage = jetImage;
  }

  return iplImage;
}

/*
 * Deprecated function.
 */
void Saliency::Symmetry::showScaleMap(float *map, int width, int height, int sc) {
  normMap0to1(map);
  //IplImage *scaleI = cvCreateImageHeader(cvSize(width, height), IPL_DEPTH_32F, 1);
  //IplImage *scaleII = cvCreateImage(cvSize(widthSymMap, heightSymMap), IPL_DEPTH_32F, 1);
  //cvSetData(scaleI, sMapScale, width*sizeof(float));
  //cvResize(scaleI, scaleII);
	
  IplImage *jetImage = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 3);
  char *jetData = jetImage->imageData;
  int jetI,ii,iii;
  for(int y=0; y<height; y++) {
    for(int x=0; x<width; x++) {
      ii = y*jetImage->widthStep + jetImage->nChannels*x;
      iii = y*width + x;
      jetI = (int)floor(map[iii]*(color->jetSize-1));
      jetData[ii] = (char)rint(color->jetU8[3*jetI+2]);
      jetData[ii+1] = (char)rint(color->jetU8[3*jetI+1]);
      jetData[ii+2] = (char)rint(color->jetU8[3*jetI]);
    }
  }
  char winName[100];
  sprintf(winName, "scale%d", sc);
  cvShowImage(winName, jetImage);
  cvWaitKey(0);
}

/*
 * Inline function to get cosine values from the look-up table
 */
inline float Saliency::Symmetry::cosFn(float x) {
  return(cosAr[(int)(cosArSize*x/(4*M_PI))  + cosArSize]);
}

/*
 * Inline function to get sine values from the look-up table
 */
inline float Saliency::Symmetry::sinFn(float x) {
  return(cosAr[(int)(sinArSize*x/(4*M_PI))  + sinArSize]);
}

/*
 * Inline function to get logarithmic values from the look-up table
 */
inline float Saliency::Symmetry::logFn(float x) {
  // Input value from 1 to 1+sqrt(72)   1->0    1+sqrt(72)->logArSize-1
  //float v = logAr[ (int)((x-1)/sqrt(72)*(logArSize-1)) ];

  return( logAr[ (int)((x-1)/sqrt(72.0)*(logArSize-1)) ] ); 
}

/*
 * Clears the member variable strongestRadiusMap.
 */
void Saliency::Symmetry::clearRadiusMaps() {
  if(strongestRadiusMaps.size()>0) {
    for(unsigned int i=0; i<strongestRadiusMaps.size(); i++)
      delete strongestRadiusMaps[i];
    strongestRadiusMaps.clear();
  }
}

/*
 * Returns all proto-objects in the saliency map. In order to be able to use this function,
 * the strongest radii need to have been calculated (set calcStrongestRadius to true). The 
 * proto-objects are the image regions that contributed to the symmetry values in the map.
 * They are calculated in six steps. Step 1 finds all local maxima in the symmetry map that
 * have a symemtry value higher than a proportion (seedT) of the global maximum. Step 2 finds 
 * symmetryical regions that connect to these local maxima. This is done by region growing. 
 * All points connected to the local maxima with symmetry values higher than a proportion 
 * (roiT) of the global maximum are added to the regions. In step 3 the labels are
 * sorted out, so that each symmetry area gets a unique id. Step 4 the strongest contributing
 * radii for each point in the symmetry areas. This is then used in step 5 to mark the areas 
 * of the proto-objects. For each point in the symmetry areas, a circles is drawn in the object
 * map with the radius of the strongest contributing radius to the symmetry value for that point.
 * Finally in step 6, the bounding box of the proto-objects is calculated. 
 * This method has been used in (Kootstra \em et \em al, IROS 2009) to find landmarks in images.
 * There the areas of the proto-objects were described using histogram of gradients, similar
 * to SIFT.
 *
 * @return Returns a vector with pointers to object maps. Each object map is of type ObjectMap.
 */
vector<ObjectMap *>Saliency::Symmetry::getObjectMap() {
  int width = widthSymMap;
  int height = heightSymMap;
	
  // Create the object map
  uchar *objectMap = new uchar[width*height]; // TODO: better to have it an int, but float for easy displaying
  for(int k=0; k<width*height; k++)
    objectMap[k] = 0;

  ////////////////////////////
  // Step 1: Find the local maxima in the overall symmetry map
  // symMap is probabily normalized by the IK method. The highest value is therefore != 1
  // Find global maximum
  vector<ObjectPoint> seeds;
  seeds.reserve(1000); // TODO: What is a reasonable number?

  // Find maximum value in the map
  float maxV;
  if(!objThresholdAbsolute) {
    maxV=-1;
    for(int i=0; i<width*height; i++) {
      if(symMap[i] > maxV)
	maxV = symMap[i];
    }
  }
  else {
    maxV=1;
  }

  // Find local maxima
  int i;
  ObjectPoint objectPoint;
  int nrLabels;
  int label = 1;
  for(int y=0; y<height; y++) {
    for(int x=0; x<width; x++) {
      i = y*width + x;
      if(symMap[i] >= maxV*seedT) {
	if(localMaximum(symMap[i], symMap, x, y, width, height)) {
	  // Add i to the seed vector
	  objectPoint.x = x;
	  objectPoint.y = y;
	  objectPoint.label = label;
	  seeds.push_back(objectPoint);
	  objectMap[toI(x, y, width)] = label;
	  label++;
	}
      }
    }
  }
  nrLabels = label-1;

  vector<ObjectPoint> originalSeeds = seeds;

  ////////////////////////////
  // Step 2: Grow regions from these maxima, and mark the label.
	
  vector<LabelLink> labelLinks;
  LabelLink link;
  int labelA, labelB;
  int xIn, yIn;

  ObjectPoint pntIn, pnt;

  // The threshold is now relative to the global maximum. 
  // TODO: set the threshold relative to the original seed (the local maximum)
  unsigned int k=0;
  while(k < seeds.size()) {
    pnt = seeds[k];
    k++;
    // Process all neighbors
    xIn = pnt.x-1; 
    yIn = pnt.y; 
    if(objectMap[toI(xIn, yIn, width)]==0 && xIn>=0 && symMap[toI(xIn, yIn, width)] >=  maxV*roiT) {
      pntIn.x = xIn; 
      pntIn.y = yIn; 
      pntIn.label = pnt.label; 
      objectMap[toI(xIn, yIn, width)] = pnt.label; 
      seeds.push_back(pntIn);
    } 
    else if(objectMap[toI(xIn, yIn, width)]>0) {
      labelA = pnt.label;
      labelB = objectMap[toI(xIn, yIn, width)];
      if(labelA>labelB) {
	link.labelA = labelA;
	link.labelB = labelB;
	labelLinks.push_back(link);
      } else if(labelB>labelA) {
	link.labelA = labelB;
	link.labelB = labelA;
	labelLinks.push_back(link);
      }
      // else, do not log, since labels are the same
    }

    xIn = pnt.x+1; 
    yIn = pnt.y; 
    if(objectMap[toI(xIn, yIn, width)]==0 && xIn<width && symMap[toI(xIn, yIn, width)] >=  maxV*roiT) {
      pntIn.x = xIn; 
      pntIn.y = yIn; 
      pntIn.label = pnt.label; 
      objectMap[toI(xIn, yIn, width)] = pnt.label; 
      seeds.push_back(pntIn);
    } 
    else if(objectMap[toI(xIn, yIn, width)]>0) {
      labelA = pnt.label;
      labelB = objectMap[toI(xIn, yIn, width)];
      if(labelA>labelB) {
	link.labelA = labelA;
	link.labelB = labelB;
	labelLinks.push_back(link);
      } else if(labelB>labelA) {
	link.labelA = labelB;
	link.labelB = labelA;
	labelLinks.push_back(link);
      }
      // else, do not log, since labels are the same
    }

    xIn = pnt.x; 
    yIn = pnt.y-1; 
    if(objectMap[toI(xIn, yIn, width)]==0 && yIn>=0 && symMap[toI(xIn, yIn, width)] >=  maxV*roiT) {
      pntIn.x = xIn; 
      pntIn.y = yIn; 
      pntIn.label = pnt.label; 
      objectMap[toI(xIn, yIn, width)] = pnt.label; 
      seeds.push_back(pntIn);
    } 
    else if(objectMap[toI(xIn, yIn, width)]>0) {
      labelA = pnt.label;
      labelB = objectMap[toI(xIn, yIn, width)];
      if(labelA>labelB) {
	link.labelA = labelA;
	link.labelB = labelB;
	labelLinks.push_back(link);
      } else if(labelB>labelA) {
	link.labelA = labelB;
	link.labelB = labelA;
	labelLinks.push_back(link);
      }
      // else, do not log, since labels are the same
    }

    xIn = pnt.x; 
    yIn = pnt.y+1; 
    if(objectMap[toI(xIn, yIn, width)]==0 && yIn<height && symMap[toI(xIn, yIn, width)] >=  maxV*roiT) {
      pntIn.x = xIn; 
      pntIn.y = yIn; 
      pntIn.label = pnt.label; 
      objectMap[toI(xIn, yIn, width)] = pnt.label; 
      seeds.push_back(pntIn);
    } 
    else if(objectMap[toI(xIn, yIn, width)]>0) {
      labelA = pnt.label;
      labelB = objectMap[toI(xIn, yIn, width)];
      if(labelA>labelB) {
	link.labelA = labelA;
	link.labelB = labelB;
	labelLinks.push_back(link);
      } else if(labelB>labelA) {
	link.labelA = labelB;
	link.labelB = labelA;
	labelLinks.push_back(link);
      }
      // else, do not log, since labels are the same
    }
  }
	
  delete[] objectMap;

  ////////////////////////////
  // Step 3: Sort out the labels
	
  sortOutLabels(seeds, width, height, labelLinks, nrLabels);

  ////////////////////////////
  // Step 4: Find the strongest contributing radius per RoI pixel 

  float strongestRadius=0;
  float strongestValue=-1;
  float value;
  float rescale;
  int scaleWidth;

  for(unsigned int k=0; k<seeds.size(); k++) {
    // Find the maximum contributing radius for this point from the different scale maps
    strongestValue = -1;
    strongestRadius = 0;
    pnt = seeds[k];
    for(unsigned int s=0; s<strongestRadiusMaps.size(); s++) {
      rescale = pow(2.0,(double)-strongestRadiusMaps[s]->scale); // Relative to resolution of symMap (first scale)
      scaleWidth = strongestRadiusMaps[s]->width;
      value = strongestRadiusMaps[s]->valueMap[(int)rint(rescale*pnt.y) * scaleWidth + (int)rint(rescale*pnt.x)];

      if(value > strongestValue) {
	strongestValue = value;
	// Set absolute radius relative to first scale (is to resolution symMap)
	// Calculate the radius. Take the sqrt and devide by 2, since the square diameter is stored in radiusMap (r = sqrt(d)/2)
	strongestRadius = pow(2.0,(double)strongestRadiusMaps[s]->scale) * sqrt( strongestRadiusMaps[s]->radiusMap[(int)(rescale*pnt.y) * scaleWidth + (int)(rescale*pnt.x)] )/2;
      }
    }
    //seeds[k].r = sqrt(strongestRadius)/2; // strongestRadius is the square of the diameter: (r = sqrt(d)/2)
    seeds[k].r = strongestRadius; // strongestRadius is the square of the diameter: (r = sqrt(d)/2)
    seeds[k].v = strongestValue; // strongestRadius is the square of the diameter: (r = sqrt(d)/2)
  }

  ////////////////////////////
  // Step 5: Label all object points by placing a circle with the given radius
  // and calculated bounding box and center.
  // Center calculation based on the objectMap is instable. Based on the symmetry region is 
  // much more robust. TODO: test the effect of the weighting with the symmetry value
  // Not weighing the center seems to be more robust and give better depth information

  vector<ObjectMap *>objectMaps;
  for(int label=1; label<nrLabels+1; label++) {
    ObjectMap *objMap = new ObjectMap;

    IplImage *objectMapIpl = cvCreateImageHeader(cvSize(width,height), IPL_DEPTH_8U, 1);
    objMap->map = new uchar[width*height];
    cvSetData(objectMapIpl, objMap->map, width*sizeof(uchar));
    cvZero(objectMapIpl);

    int x0=width, y0=height, x1=0, y1=0;
    float xTot=0.0, yTot=0.0;
    float xyNorm=0.0;
    for(unsigned int k=0; k<seeds.size(); k++) {
      if(seeds[k].label == label) {
	cvCircle(objectMapIpl, cvPoint(seeds[k].x, seeds[k].y), (int)rint(seeds[k].r), cvScalar(255), -1); //cvScalar(seeds[k].label),-1); 
	if(max(0, seeds[k].x-(int)rint(seeds[k].r)) < x0)
	  x0 = max(0, seeds[k].x-(int)rint(seeds[k].r));
	if(max(0, seeds[k].y-(int)rint(seeds[k].r)) < y0)
	  y0 = max(0, seeds[k].y-(int)rint(seeds[k].r));
				
	if(min(width-1, seeds[k].x+(int)rint(seeds[k].r)+1) > x1)
	  x1 = min(width-1, seeds[k].x+(int)rint(seeds[k].r));
	if(min(height-1, seeds[k].y+(int)rint(seeds[k].r)) > y1)
	  y1 = min(height-1, seeds[k].y+(int)rint(seeds[k].r));
	xTot += (float)seeds[k].x;
	yTot += (float)seeds[k].y;
	xyNorm += 1;
      }
    }
    // Set the rest of the information
    objMap->width = width;
    objMap->height = height;
    objMap->x0 = x0;
    objMap->y0 = y0;
    objMap->x1 = x1;
    objMap->y1 = y1;
    objMap->x = xTot/xyNorm;
    objMap->y = yTot/xyNorm;
    objMap->scale = userScales[0];
    objectMaps.push_back(objMap);

    cvReleaseImageHeader(&objectMapIpl);
  }

  ////////////////////////////
  // Step 6: Calculate the boundaries in the original scale 
  float rescale2 = pow(2.0, -(double)userScales[0]);
  for(unsigned int objI=0; objI<objectMaps.size(); objI++) {
    objectMaps[objI]->x0sc = (int)floor((float)objectMaps[objI]->x0/rescale2);
    objectMaps[objI]->y0sc = (int)floor((float)objectMaps[objI]->y0/rescale2);
    objectMaps[objI]->x1sc = (int)ceil(((float)objectMaps[objI]->x1+1)/rescale2)-1;
    objectMaps[objI]->y1sc = (int)ceil(((float)objectMaps[objI]->y1+1)/rescale2)-1;
    objectMaps[objI]->xsc = objectMaps[objI]->x/rescale2;
    objectMaps[objI]->ysc = objectMaps[objI]->y/rescale2;
  }

  return objectMaps;
}

/*
 * Checks if the point (x,y) is a local maximum in a rectangle with size 2*seedR+1.
 * @return Returns a boolean indicating whether the point is a local maximum
 */
inline bool Saliency::Symmetry::localMaximum(float value, float *symMap, int x, int y, int width, int height) {
  //seedR=1;
  for(int y2=max(0,y-seedR); y2<min(height, y+seedR+1); y2++) {
    for(int x2=max(0,x-seedR); x2<min(width, x+seedR+1); x2++) {
      if(symMap[y2*width+x2]>value && !(x2==x && y2==y))
	return(false);
    }
  }
  return(true);
}

/*
 * Sorts out labels for the proto-object calculation.
 * @see Used in getObjectMap
 */
void Saliency::Symmetry::sortOutLabels(vector<ObjectPoint> &seeds, int width, int height, vector<LabelLink> &labelLinks, int &nrLabels) {
  // Construct table with all links
  bool **linkTable; 
  linkTable = new bool*[nrLabels];
  for(int i=0; i<nrLabels; i++)
    linkTable[i] = new bool[nrLabels];

  for(int i=0; i<nrLabels; i++) {
    for(int j=0; j<nrLabels; j++) {
      linkTable[i][j] = false;
    }
  }
  int labelA, labelB;
  for(unsigned int i=0; i<labelLinks.size(); i++) {
    labelA = labelLinks[i].labelA;
    labelB = labelLinks[i].labelB;
    linkTable[labelA-1][labelB-1] = true;
    linkTable[labelB-1][labelA-1] = true;
  }


  vector< vector<uchar> > labelGroups;
  vector<uchar> group;
  bool *added = new bool[nrLabels];
  for(int label=1; label<nrLabels+1; label++) 
    added[label-1] = false;

  for(int label=1; label<nrLabels+1; label++) {
    if(!added[label-1]) {
      group.clear();

      // Mark as added
      added[label-1] = true;
      // Add to group
      group.push_back(label);
      // Add all members of the row recursively
      addRowLabels(linkTable, label, nrLabels, group, added);

      labelGroups.push_back(group);
    }
  }

  // construct relabel array
  uchar *reLabels = new uchar[nrLabels+1];
  uchar reLabel;
  for(int label=1; label<nrLabels+1; label++) {
    int i=0;
    reLabel=0;
    while(reLabel==0) {
      for(unsigned int k=0; k<labelGroups[i].size(); k++) {
	if(labelGroups[i][k]==(uchar)label) {
	  reLabel = (uchar)(i+1);
	  break;
	}
      }
      i++;
    }
    reLabels[label] = reLabel;
  }

  // Relabel the seeds vector
  for(unsigned int i=0; i<seeds.size(); i++)
    seeds[i].label =  reLabels[ seeds[i].label ];

  for(int i=0; i<nrLabels; i++)
    delete[] linkTable[i];
  delete[] linkTable;
  delete[] added;
  delete[] reLabels;
	
  nrLabels = labelGroups.size();
}

/*
 * Function use to sort out the labels of the proto-objects.
 * @see Used in sortOutLabels
 */

void Saliency::Symmetry::addRowLabels(bool **linkTable, uchar label, int nrLabels, vector<uchar> &group, bool *added) {

  for(int i=0; i<nrLabels; i++) {
    if(linkTable[label-1][i] && !added[i]) {
      // Mark as added
      added[i] = true;
      // Add to group
      group.push_back(i+1);
      // Add all members of the row
      addRowLabels(linkTable, i+1, nrLabels, group, added);
    }
  }
}

/*
 * @return Returns the scale of the symmetry-saliency map. Scale 0 is equal to the original
 *         scale of the input image. Next scales are subsequently down scaled by a factor two.
 */
int Saliency::Symmetry::getScale() {
  return userScales[0];
}

#undef TIMING
