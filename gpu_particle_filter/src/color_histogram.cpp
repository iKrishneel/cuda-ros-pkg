//  Created by Chaudhary Krishneel on 3/28/14.
//  Copyright (c) 2014 Chaudhary Krishneel. All rights reserved.
//

#include <gpu_particle_filter/color_histogram.h>

ColorHistogram::ColorHistogram() :
    h_bins(10), s_bins(10) {
   
}

void ColorHistogram::computeHistogram(
    cv::Mat &src, cv::Mat &hist, bool isNormalized) {
    cv::Mat hsv;
    cv::cvtColor(src, hsv, CV_BGR2HSV);
    int histSize[] = {this->h_bins, this->s_bins};
    float h_ranges[] = {0, 180};
    float s_ranges[] = {0, 256};
    const float* ranges[] = {h_ranges, s_ranges};
    int channels[] = {0, 1};
    cv::calcHist(
       &hsv, 1, channels, cv::Mat(), hist, 2, histSize, ranges, true, false);
    if (isNormalized) {
       cv::normalize(hist, hist, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());
    }
}

double ColorHistogram::computeHistogramDistances(
    cv::Mat &hist, std::vector<cv::Mat> *hist_MD , cv::Mat *h_D ) {
    double sum = 0.0;
    double max_distance = FLT_MAX;
    if (h_D != NULL) {
       sum = static_cast<double>(
          cv::compareHist(hist, *h_D, CV_COMP_BHATTACHARYYA));
    } else if (hist_MD->size() > 0) {
       int i;
#ifdef _OPENMP
#pragma omp parallel for private(i) shared(sum, max_distance)
#endif
       for (i = 0; i < hist_MD->size(); i++) {
          double d = static_cast<double>(
             cv::compareHist(hist, (*hist_MD)[i], CV_COMP_BHATTACHARYYA));
          if (d < max_distance) {
#ifdef _OPENMP
#pragma omp critical
#endif
             max_distance = static_cast<double>(d);
          }
       }
       sum = static_cast<double>(max_distance);
    }
    return sum;
}

