
#ifndef _GPU_PARTICLE_FILTER_H_
#define _GPU_PARTICLE_FILTER_H_

#include <ros/ros.h>
#include <ros/console.h>
#include <gpu_particle_filter/particle_filter_kernel.h>

#include <image_geometry/pinhole_camera_model.h>
#include <cv_bridge/cv_bridge.h>

#include <geometry_msgs/PolygonStamped.h>
#include <jsk_recognition_msgs/Rect.h>
#include <sensor_msgs/Image.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#include <boost/thread/mutex.hpp>

class ParticleFilterGPU {

 private:
    cv::Rect_<int> screen_rect_;
    bool tracker_init_;
    int width_;
    int height_;
    int block_size_;
    int downsize_;

    cv::Mat dynamics;
    cv::RNG random_num_;
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
    virtual void imageCB(const sensor_msgs::Image::ConstPtr &);
    virtual void screenPtCB(const geometry_msgs::PolygonStamped &);
};

#endif  // _GPU_PARTICLE_FILTER_H_
