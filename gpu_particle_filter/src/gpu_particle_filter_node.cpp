// Copyright (C) 2015 by Krishneel Chaudhary,
// JSK Lab, The University of Tokyo

#include <gpu_particle_filter/gpu_particle_filter.h>

ParticleFilterGPU::ParticleFilterGPU():
    block_size_(8), downsize_(1),
    tracker_init_(false), threads_(8) {
    this->gpu_init_ = true;
    this->onInit();
}

void ParticleFilterGPU::onInit() {
    this->subscribe();
    this->pub_image_ = pnh_.advertise<sensor_msgs::Image>(
        "target", 1);
}

void ParticleFilterGPU::subscribe() {
    this->sub_screen_pt_ = this->pnh_.subscribe(
        "input_screen", 1, &ParticleFilterGPU::screenPtCB, this);
    this->sub_image_ = this->pnh_.subscribe(
        "image", 1, &ParticleFilterGPU::imageCB, this);
}

void ParticleFilterGPU::unsubscribe() {
    this->sub_image_.shutdown();
}

void ParticleFilterGPU::screenPtCB(
    const geometry_msgs::PolygonStamped &screen_msg) {
    int x = screen_msg.polygon.points[0].x;
    int y = screen_msg.polygon.points[0].y;
    int width = screen_msg.polygon.points[1].x - x;
    int height = screen_msg.polygon.points[1].y - y;
    this->screen_rect_ = cv::Rect_<int>(
        x/downsize_, y/downsize_, width/downsize_, height/downsize_);
    if (width > this->block_size_ && height > this->block_size_) {
        this->tracker_init_ = true;
    } else {
        ROS_WARN("-- Selected Object Size is too small... Not init tracker");
    }

    gpu_init_ = true;
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
    if (image.empty()) {
        ROS_ERROR("EMPTY INPUT IMAGE");
        return;
    }
    if (downsize_ > 1) {
        cv::resize(image, image, cv::Size(image.cols/this->downsize_,
                                          image.rows/this->downsize_));
    }

    if (tracker_init_) {
       particleFilterGPU(image, screen_rect_, gpu_init_);
    } else {
        ROS_ERROR("THE TRACKER IS NOT INITALIZED");
    }
    
    cv_bridge::CvImagePtr pub_msg(new cv_bridge::CvImage);
    pub_msg->header = image_msg->header;
    pub_msg->encoding = sensor_msgs::image_encodings::BGR8;
    pub_msg->image = image.clone();
    this->pub_image_.publish(pub_msg);
    
    cv::namedWindow("Tracking", cv::WINDOW_NORMAL);
    cv::imshow("image", image);
    cv::waitKey(3);
}

int main(int argc, char *argv[]) {
    ros::init(argc, argv, "gpu_particle_filter");
    ParticleFilterGPU pfg;
    ros::spin();
    return 0;
}
