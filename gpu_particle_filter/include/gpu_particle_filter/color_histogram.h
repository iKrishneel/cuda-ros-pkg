//  Created by Chaudhary Krishneel on 3/28/14.
//  Copyright (c) 2014 Chaudhary Krishneel. All rights reserved.

#ifndef _COLOR_HISTOGRAM_H_
#define _COLOR_HISTOGRAM_H_

#include <opencv2/opencv.hpp>
#include <omp.h>

#include <vector>

class ColorHistogram {
 private:
    int h_bins;
    int s_bins;
   
 public:
    ColorHistogram();
    void computeHistogram(
       cv::Mat &, cv::Mat &, bool CV_DEFAULT(true));
    double computeHistogramDistances(
       cv::Mat &, std::vector<cv::Mat> *hist_MD CV_DEFAULT(NULL),
       cv::Mat *h_D CV_DEFAULT(NULL));
};

#endif /* defined(_COLOR_HISTOGRAM_H_) */
