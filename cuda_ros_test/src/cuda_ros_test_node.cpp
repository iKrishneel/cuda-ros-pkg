
#include <cuda_ros_test/cuda_ros_test.h>

CudaRosTest::CudaRosTest() {    
    this->onInit();
}

void CudaRosTest::onInit() {
    this->subscribe();
    this->pub_image_ = pnh_.advertise<sensor_msgs::Image>(
        "target", 1);
}

void CudaRosTest::subscribe() {
    this->sub_image_ = this->pnh_.subscribe(
        "image", 1, &CudaRosTest::imageCB, this);
}

void CudaRosTest::unsubscribe() {
    this->sub_image_.shutdown();
}

void CudaRosTest::imageCB(
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

    // cv::cvtColor(image, image, CV_BGR2GRAY);
    // boxFilter(image, 23);
    // boxFilter2D(image, 3);
    boxFilterMan(image, 12);
    // cv::cvtColor(image, image, CV_GRAY2BGR);

    cv::namedWindow("image", cv::WINDOW_NORMAL);
    cv::imshow("image", image);
    cv::waitKey(3);
    
    cv_bridge::CvImagePtr pub_msg(new cv_bridge::CvImage);
    pub_msg->header = image_msg->header;
    pub_msg->encoding = sensor_msgs::image_encodings::BGR8;
    pub_msg->image = image.clone();
    this->pub_image_.publish(pub_msg);
}


void CudaRosTest::boxFilterCPU(
    cv::Mat &image, const int size) {
    for (int j = 0; j < image.rows; j++) {
        for (int i = 0; i < image.cols; i++) {
            // if ((j > size && j < image.rows - size) &&
            //     (i > size && i < image.cols - size)) {
            if (j > size && i > size && j < image.rows - size
                && i < image.cols - size) {
                int val = 0;
                int icounter = 0;
                for (int y = -size; y < size + 1; y++) {
                    for (int x = -size; x < size + 1; x++) {
                        val += (int)image.at<uchar>(j + y, i + x);
                        icounter++;
                    }
                }
                image.at<uchar>(j, i) = val / icounter;
            }
        }
    }
}

int main(int argc, char *argv[]) {
    ros::init(argc, argv, "gpu_particle_filter");
    CudaRosTest pfg;
    ros::spin();
    return 0;
}

