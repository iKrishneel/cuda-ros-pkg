
#ifndef _GPU_PARTICLE_FILTER_H_
#define _GPU_PARTICLE_FILTER_H_

#include <ros/ros.h>
#include <ros/console.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <image_geometry/pinhole_camera_model.h>
#include <cv_bridge/cv_bridge.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>

#include <boost/thread/mutex.hpp>

#include <geometry_msgs/PolygonStamped.h>
#include <jsk_recognition_msgs/Rect.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointCloud2.h>

#include <gpu_particle_filter/particle_filter.h>
#include <gpu_particle_filter/particle_filter_kernel.h>
#include <gpu_particle_filter/color_histogram.h>
#include <omp.h>

class ParticleFilterGPU: public ParticleFilter,
                         public ColorHistogram {

 private:
    std::vector<cv::Mat> reference_object_histogram_;
    std::vector<cv::Mat> reference_background_histogram_;

    cv::Rect_<int> screen_rect_;
    bool tracker_init_;
    int width_;
    int height_;
    int block_size_;
    int downsize_;

    int hbins;
    int sbins;

    cv::Mat dynamics;
    std::vector<Particle> particles;
    cv::RNG random_num_;
    std::vector<cv::Point2f> particle_prev_position;
    cv::Mat prev_frame_;

    bool gpu_init_;
    
 protected:
    virtual void onInit();
    virtual void subscribe();
    virtual void unsubscribe();

    boost::mutex lock_;
    ros::NodeHandle pnh_;
    ros::Subscriber sub_image_;
    ros::Subscriber sub_screen_pt_;
    ros::Publisher pub_image_;
    unsigned int threads_;
    
  public:
    ParticleFilterGPU();
    virtual void imageCB(
        const sensor_msgs::Image::ConstPtr &);
    virtual void screenPtCB(
        const geometry_msgs::PolygonStamped &);

    void initializeTracker(
        const cv::Mat &, cv::Rect &);

    void runObjectTracker(
        cv::Mat *image, cv::Rect &rect);
    std::vector<cv::Mat> imagePatchHistogram(
        cv::Mat &);
    std::vector<cv::Mat> particleHistogram(
        cv::Mat &, std::vector<Particle> &);
    std::vector<double> colorHistogramLikelihood(
        std::vector<cv::Mat> &);
    void roiCondition(cv::Rect &, cv::Size);

    void multiResolutionColorContrast(
        cv::Mat &, const cv::Mat &, const int);
};

#endif  // _GPU_PARTICLE_FILTER_H_
