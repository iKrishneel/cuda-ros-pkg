
#ifndef _PARTICLE_FILTER_KERNEL_H_
#define _PARTICLE_FILTER_KERNEL_H_

#include <stdio.h>
#include <math.h>
#include <opencv2/opencv.hpp>

const int N = 16;
const int blocksize = 16;

void test_cuda(int *, int *);
void boxFilter(cv::Mat &, const int size);

#endif // _PARTICLE_FILTER_KERNEL_H_
