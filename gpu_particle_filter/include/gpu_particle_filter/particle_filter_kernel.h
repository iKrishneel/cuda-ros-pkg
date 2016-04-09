
#ifndef _PARTICLE_FILTER_KERNEL_H_
#define _PARTICLE_FILTER_KERNEL_H_

#include <stdio.h>
#include <math.h>

#include <curand.h>
#include <curand_kernel.h>

#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaobjdetect.hpp>

#include <gpu_particle_filter/color_histogram_kernel.h>

// filter params
#define BLOCK_SIZE 16
#define PARTICLES_SIZE 2
#define STATE_SIZE 4
#define G_SIGMA 5

// feature params
#define PATCH_SIZE 16
#define COLOR_BINS 8
#define COLOR_CHANNEL 3
#define HOG_DIM 36

// control param
#define PROBABILITY_THRESH 0.7
#define COLOR_CONTRL 0.7
#define HOG_CONTRL 0.7

void particleFilterGPU(cv::Mat &, cv::Rect &, bool &);
void gpuHist(cv::Mat, cv::Mat);

#endif // _PARTICLE_FILTER_KERNEL_H_
