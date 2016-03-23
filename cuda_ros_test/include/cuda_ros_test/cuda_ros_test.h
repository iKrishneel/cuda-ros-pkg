
#ifndef _CUDA_ROS_TEST_H_
#define _CUDA_ROS_TEST_H_

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
#include <boost/thread/mutex.hpp>

#include <geometry_msgs/PolygonStamped.h>
#include <jsk_recognition_msgs/Rect.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointCloud2.h>

#include <cuda_ros_test/particle_filter_kernel.h>
#include <omp.h>

class CudaRosTest {

 private:
    boost::mutex lock_;
    ros::NodeHandle pnh_;
    ros::Subscriber sub_image_;
    ros::Publisher pub_image_;
    unsigned int threads_;
    
 protected:
    virtual void onInit();
    virtual void subscribe();
    virtual void unsubscribe();
    
  public:
    CudaRosTest();
    virtual void imageCB(
        const sensor_msgs::Image::ConstPtr &);

    void boxFilterCPU(cv::Mat &, const int);
};

#endif  // _CUDA_ROS_TEST_H_
