
#include <gpu_particle_filter/gpu_particle_filter.h>

ParticleFilterGPU::ParticleFilterGPU() {
    
    std::cout << "GPU TEST"  << "\n";

    char a[N] = "HELLO  \0\0\0\0\0\0";
    int b[N] = {
        15, 10, 6, 0, -11,
        1, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0
    };
    // test_cuda(a, b);
    // printf("%s\n", a);
    cv::Mat image = cv::imread("/home/krishneel/Desktop/15m.jpg", 0);
    if (image.empty()) {
        ROS_ERROR("EMPTY");
        return;
    }
    boxFilter(image, 27);
    std::cout << "Done."  << "\n";
    
    // this->onInit();
}

void ParticleFilterGPU::onInit() {
    this->subscribe();
    this->pub_image_ = pnh_.advertise<sensor_msgs::Image>(
        "target", 1);
}

void ParticleFilterGPU::subscribe() {
    this->sub_image_ = this->pnh_.subscribe(
        "image", 1, &ParticleFilterGPU::imageCB, this);
}

void ParticleFilterGPU::unsubscribe() {
    this->sub_image_.shutdown();
}

void ParticleFilterGPU::imageCB(
    const sensor_msgs::Image::ConstPtr &image_msg) {
    cv_bridge::CvImagePtr cv_ptr;
    try {
        cv_ptr = cv_bridge::toCvCopy(
            image_msg, sensor_msgs::image_encodings::BGR8);
    } catch (cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }
    cv::Mat image = cv_ptr->image.clone();
}

int main(int argc, char *argv[]) {
    ros::init(argc, argv, "gpu_particle_filter");
    ParticleFilterGPU pfg;
    ros::spin();
    return 0;
}

