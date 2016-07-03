
#include <gpu_skeletonization/gpu_skeletonization.h>

GPUSkeletonization::GPUSkeletonization() {

    int icounter = 0;
    while (icounter++ < 50) {
       cv::Mat image = cv::imread("/home/krishneel/Desktop/mbzirc/mask.png");
       cv::resize(image, image, cv::Size(1280, 960));
       std::cout << image.size()  << "\n";
    
       skeletonizationGPU(image);

       ROS_WARN("DONE");

       cv::imshow("input", image);
       cv::waitKey(30);
    }
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
