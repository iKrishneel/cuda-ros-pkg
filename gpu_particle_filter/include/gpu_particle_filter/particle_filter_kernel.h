
#ifndef _PARTICLE_FILTER_KERNEL_H_
#define _PARTICLE_FILTER_KERNEL_H_

#include <stdio.h>
#include <math.h>

#include <curand.h>
#include <curand_kernel.h>

#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>

#include <gpu_particle_filter/color_histogram_kernel.h>

const int N = 16;

#define BLOCK_SIZE 16
#define PARTICLES_SIZE 500
#define STATE_SIZE 4
#define G_SIGMA 5

#define COLOR_BINS 8
#define COLOR_CHANNEL 3

void particleFilterGPU(cv::Mat &, cv::Rect &, bool &);

void gpuHist(cv::Mat image, cv::Mat cpu_hist);

/**
 * 
 */
void test_cuda(int *, int *);
void boxFilter(cv::Mat &, const int size);
int iDivUp(int, int);
void boxFilter2D(cv::Mat &, const int);
void boxFilterMan(cv::Mat &, const int);

#endif // _PARTICLE_FILTER_KERNEL_H_
