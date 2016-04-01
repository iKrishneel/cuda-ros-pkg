//  Created by Chaudhary Krishneel on 4/9/14.
//  Copyright (c) 2014 Chaudhary Krishneel. All rights reserved.

#ifndef _PARTICLE_FILTER_H_
#define _PARTICLE_FILTER_H_

#include <opencv2/opencv.hpp>
#include <omp.h>

struct Particle{
    double x;
    double y;
    double dx;
    double dy;
};

class ParticleFilter {
   
#define PI 3.14159265358979323846
#define SIGMA 10.0
#define NUM_PARTICLES 500
#define NUM_STATE 4
   
 public:
   ParticleFilter();
    cv::Mat state_transition();
    std::vector<Particle> initialize_particles(
      cv::RNG &, double, double, double, double);
    std::vector<Particle> transition(
       std::vector<Particle> &, cv::Mat &, cv::RNG &);
    double evaluate_gaussian(double, double);
    std::vector<double> normalizeWeight(
       std::vector<double> &z);
    double gaussianNoise(
       double, double);
    std::vector<double> cumulativeSum(
        std::vector<double> &);
    void reSampling(
      std::vector<Particle> &, std::vector<Particle> &,
      std::vector<double> &);
    Particle meanArr(
       std::vector<Particle> &);
    void printParticles(
       cv::Mat &, std::vector<Particle> &);

 protected:
    unsigned int threads_;
};


#endif  // _PARTICLE_FILTER_H_

