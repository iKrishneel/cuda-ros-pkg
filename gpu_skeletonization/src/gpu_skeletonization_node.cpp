
#include <gpu_skeletonization/gpu_skeletonization.h>

GPUSkeletonization::GPUSkeletonization() {

    cv::Mat image = cv::imread("/home/krishneel/Desktop/mbzirc/mask.png");
    std::cout << image.size()  << "\n";

    skeletonizationGPU(image);

    ROS_WARN("DONE");
}

void GPUSkeletonization::onInit() {
   
}

void GPUSkeletonization::subscribe() {
   
}

void GPUSkeletonization::unsubscribe() {
   
}

void GPUSkeletonization::callback(
    const sensor_msgs::Image::ConstPtr &image_msg) {
   
}



int main(int argc, char *argv[]) {

    ros::init(argc, argv, "gpu_skeletonization");
    GPUSkeletonization gpu_s;
    ros::spin();
    return 0;
}
