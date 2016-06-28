
#ifndef _PARTICLE_FILTER_KERNEL_H_
#define _PARTICLE_FILTER_KERNEL_H_

#include <stdio.h>
#include <math.h>

#include <curand.h>
#include <curand_kernel.h>

#include <cublas.h>
#include <cublas_v2.h>

#include <opencv2/opencv.hpp>

// filter params
#define BLOCK_SIZE 16
#define PARTICLES_SIZE 10
#define STATE_SIZE 4
#define G_SIGMA 5

// feature params
#define PATCH_SIZE 16
#define COLOR_BINS 16
#define COLOR_CHANNEL 3
#define HOG_DIM 36

// control param
#define PROBABILITY_THRESH 0.9
#define COLOR_CONTRL 0.7
#define HOG_CONTRL 0.7

#define HOG_CELL 8
#define HOG_BLOCK 2
#define HOG_NBINS 9
#define HOG_ANGLE 180
#define HOG_BIN_ANGLE (HOG_ANGLE/HOG_NBINS)
#define HOG_FEATURE_DIMS (HOG_BLOCK * HOG_BLOCK * HOG_NBINS)

void particleFilterGPU(cv::Mat &, cv::Rect &, bool &);
void gpuHist(cv::Mat, cv::Mat);

#endif // _PARTICLE_FILTER_KERNEL_H_
