
#ifndef _COLOR_HISTOGRAM_KERNEL_H_
#define _COLOR_HISTOGRAM_KERNEL_H_

#include <opencv2/opencv.hpp>

#include <iostream>
#include <math.h>
#include <stdio.h>

#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaobjdetect.hpp>

#include <gpu_particle_filter/particle_filter.h>

struct cuPFFeatures {
    cv::Mat color_hist;
    cv::Mat hog_hist;
};

bool cuCreateParticlesFeature(cuPFFeatures &features, const cv::Mat &img,
                            const std::vector<Particle> &particles,
                            const int downsize);
std::vector<float> cuHistogramLikelihood(
    const std::vector<Particle> &particles,
    const std::vector<Particle> &particles_, 
    cv::Mat &image, const cuPFFeatures features,
    const cuPFFeatures prev_features);

// cuPFFeatures reference_features;
const int threads_ = 8;

#endif // _COLOR_HISTOGRAM_KERNEL_H_
