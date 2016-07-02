
#ifndef _SKELETONIZATION_KERNEL_H_
#define _SKELETONIZATION_KERNEL_H_

#include <stdio.h>
#include <math.h>

#include <curand.h>
#include <curand_kernel.h>

#include <cublas.h>
#include <cublas_v2.h>

#include <opencv2/opencv.hpp>

#define GRID_SIZE 16
enum {
    EVEN, ODD };

void skeletonizationGPU(cv::Mat);


#endif   // _SKELETONIZATION_KERNEL_H_
