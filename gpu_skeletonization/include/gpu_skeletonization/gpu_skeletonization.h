
#ifndef _GPU_SKELETONIZATION_H_
#define _GPU_SKELETONIZATION_H_

#include <ros/ros.h>
#include <ros/console.h>
#include <cv_bridge/cv_bridge.h>

#include <geometry_msgs/PolygonStamped.h>
#include <jsk_recognition_msgs/Rect.h>
#include <sensor_msgs/Image.h>

#include <image_geometry/pinhole_camera_model.h>
#include <gpu_skeletonization/skeletonization_kernel.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

class GPUSkeletonization {

 private:
    
   
 protected:
    void onInit();
    void subscribe();
    void unsubscribe();
  
    ros::NodeHandle pnh_;
    ros::Subscriber sub_image_;
    ros::Publisher pub_image_;
   
 public:
    GPUSkeletonization();
    void callback(const sensor_msgs::Image::ConstPtr &);
};


#endif  // _GPU_SKELETONIZATION_H_
