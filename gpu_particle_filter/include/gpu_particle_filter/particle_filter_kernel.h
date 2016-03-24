
#ifndef _PARTICLE_FILTER_KERNEL_H_
#define _PARTICLE_FILTER_KERNEL_H_

#include <stdio.h>
#include <math.h>

#include <curand.h>
#include <curand_kernel.h>

#include <opencv2/opencv.hpp>

const int N = 16;
const int BLOCK_SIZE = 16;
#define PARTICLES_SIZE 50
#define STATE_SIZE 4

void particleFilterGPU(cv::Mat &, cv::Rect &, bool &);

/**
 * 
 */
void test_cuda(int *, int *);
void boxFilter(cv::Mat &, const int size);
int iDivUp(int, int);
void boxFilter2D(cv::Mat &, const int);
void boxFilterMan(cv::Mat &, const int);

#endif // _PARTICLE_FILTER_KERNEL_H_
