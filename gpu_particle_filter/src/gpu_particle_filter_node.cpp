// Copyright (C) 2015 by Krishneel Chaudhary,
// JSK Lab, The University of Tokyo

#include <gpu_particle_filter/gpu_particle_filter.h>

ParticleFilterGPU::ParticleFilterGPU():
    block_size_(8), hbins(10), sbins(12), downsize_(1),
    tracker_init_(false), threads_(8) {

    gpu_init_ = true;

    cv::Size wsize = cv::Size(16/downsize_, 16/downsize_);
    cv::Size bsize = cv::Size(16/downsize_, 16/downsize_);
    cv::Size csize = cv::Size(8/downsize_, 8/downsize_);
    this->hog_ = cv::cuda::HOG::create(wsize, bsize, csize);
    
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
    return;

    
    if (this->tracker_init_) {
        ROS_INFO("Initializing Tracker");
        this->initializeTracker(image, this->screen_rect_);
        this->tracker_init_ = false;
        ROS_INFO("Tracker Initialization Complete");

        this->prev_frame_ = image(screen_rect_).clone();
        // cv::Mat results;
        // this->intensityCorrelation(results, image, this->prev_frame_);

        // cv::resize(prev_frame_, prev_frame_,
        //        cv::Size(prev_frame_.cols/2, prev_frame_.rows/2));
        
    }
    /*
    if (this->screen_rect_.width > this->block_size_) {
        this->runObjectTracker(&image, this->screen_rect_);
    } else {
        ROS_ERROR_ONCE("THE TRACKER IS NOT INITALIZED");
    }
   
    /*
    if (!prev_frame_.empty()) {
        multiResolutionColorContrast(image, prev_frame_, 3);
        cv::namedWindow("Template", cv::WINDOW_NORMAL);
        cv::imshow("Template", prev_frame_);
    }
    */    
    
    bool run_this = false;
    if (!prev_frame_.empty() && run_this) {
        std::clock_t start;
        double duration;
        start = std::clock();

        cv::Mat results;
        // this->intensityCorrelation(results, image,
        // this->prev_frame_);

        std::cout << "Starting..."  << "\n";
        this->hogCorrelation(results, image, prev_frame_);

        duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
        std::cout<<"printf: "<< duration <<'\n';        
        cv::namedWindow("Results", cv::WINDOW_NORMAL);
        cv::imshow("Results", results);
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

void ParticleFilterGPU::initializeTracker(
    const cv::Mat &image, cv::Rect &rect) {
    this->random_num_ = cv::RNG();
    this->dynamics = this->state_transition();
    this->particles_ = this->initialize_particles(
       this->random_num_, rect.x , rect.y,
       rect.x + rect.width, rect.y + rect.height);
    
    this->createParticlesFeature(this->reference_features_, image, particles_);   
    
    /*
    cv::Mat object_region = image(rect).clone();
    cv::Mat scene_region = image.clone();
    cv::rectangle(scene_region, rect, cv::Scalar(0, 0, 0), CV_FILLED);
    this->reference_object_histogram_ = this->imagePatchHistogram(
       object_region);
    this->reference_background_histogram_ = this->imagePatchHistogram(
       scene_region);
    this->particle_prev_position.clear();
    for (int i = 0; i < NUM_PARTICLES; i++) {
        this->particle_prev_position.push_back(
           cv::Point2f(this->particles_[i].x, this->particles_[i].y));
    }
    */
    
    this->prev_frame_ = image.clone();
    this->width_ = rect.width;
    this->height_ = rect.height;
}

bool ParticleFilterGPU::createParticlesFeature(
    PFFeatures &features, const cv::Mat &img,
    const std::vector<Particle> &particles) {
    if (img.empty() || particles.empty()) {
        return false;
    }

    const int LENGHT = static_cast<int>(particles.size());
    const int dim = 8/downsize_;
    cv::Mat image;
    image = img.clone();
    cv::cvtColor(img, image, CV_BGR2HSV);
    const int bin_size = 16;

    // cv::Mat histogram = cv::Mat(LENGHT, bin_size * 6, CV_32F);
    cv::Mat histogram[LENGHT];
    cv::Mat hog_descriptors;
    cv::Rect_<int> tmp_rect[LENGHT];
        
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads_)
#endif
    for (int i = 0; i < particles.size(); i++) {
        cv::Rect_<int> rect = cv::Rect_<int>(particles[i].x - dim/2,
                                             particles[i].y - dim/2,
                                             dim, dim);
        
        this->roiCondition(rect, image.size());
        cv::Mat roi = image(rect).clone();
        cv::Mat part_hist;
        this->getHistogram(part_hist, roi, bin_size, 3, false);

        cv::Point2i s_pt = cv::Point2i((particles[i].x - dim),
                                       (particles[i].y - dim));        
        
        for (int j = 0; j < 2; j++) {
            for (int k = 0; k < 2; k++) {
                cv::Rect_<int> s_rect = cv::Rect_<int>(s_pt.x, s_pt.y, dim, dim);
                this->roiCondition(s_rect, image.size());
                roi = image(s_rect).clone();
                cv::Mat hist;
                this->getHistogram(hist, roi, bin_size, 3, false);

                if (hist.cols == part_hist.cols) {
                    for (int x = 0; x < hist.cols; x++) {
                        part_hist.at<float>(0, x) += hist.at<float>(0, x);
                    }
                }
                s_pt.x += dim;
            }
            s_pt.x = particles[i].x - dim;
            s_pt.y += dim;
        }
        s_pt = cv::Point2i((particles[i].x - dim), (particles[i].y - dim));
        rect = cv::Rect_<int>(s_pt.x, s_pt.y, dim * 2, dim * 2);
        this->roiCondition(rect, image.size());
        roi = image(rect).clone();
        cv::Mat region_hist;
        this->getHistogram(region_hist, roi, bin_size, 3, false);
        // cv::normalize(part_hist, part_hist, 0, 1, cv::NORM_MINMAX, -1);
        
        cv::Mat hist = cv::Mat::zeros(
            1, region_hist.cols + part_hist.cols, CV_32F);
        for (int x = 0; x < part_hist.cols; x++) {
            hist.at<float>(0, x) += part_hist.at<float>(0, x);
        }
        for (int x = part_hist.cols; x < hist.cols; x++) {
            hist.at<float>(0, x) += region_hist.at<float>(0, x - part_hist.cols);
        }
        cv::normalize(hist, hist, 0, 1, cv::NORM_MINMAX, -1);
        histogram[i] = hist;
        
        tmp_rect[i] = rect;
    }

    cv::Mat color_hist;
    for (int i = 0; i < particles.size(); i++) {
        cv::Mat roi = img(tmp_rect[i]).clone();
        cv::cuda::GpuMat d_roi(roi);
        cv::cuda::GpuMat d_desc;
        cv::cuda::cvtColor(d_roi, d_roi, CV_BGR2GRAY);
        this->hog_->compute(d_roi, d_desc);
        cv::Mat desc;
        d_desc.download(desc);
        hog_descriptors.push_back(desc);

        color_hist.push_back(histogram[i]);
    }
    
    features.hog_hist = hog_descriptors;
    features.color_hist = color_hist;
    return true;
}

void ParticleFilterGPU::getHistogram(
    cv::Mat &histogram, const cv::Mat &image,
    const int bins, const int chanel, bool is_norm) {
    if (image.empty()) {
        return;
    }
    histogram = cv::Mat::zeros(sizeof(char), bins * chanel, CV_32F);
    int bin_range = std::ceil(256/bins);

#ifdef _OPENMP
#pragma omp parallel for num_threads(threads_)
#endif
    for (int j = 0; j < image.rows; j++) {
        for (int i = 0; i < image.cols; i++) {
            float pixel = static_cast<float>(image.at<cv::Vec3b>(j, i)[0]);
            int bin_number = static_cast<int>(std::floor(pixel/bin_range));
            histogram.at<float>(0, bin_number)++;
            
            pixel = static_cast<float>(image.at<cv::Vec3b>(j, i)[1]);
            bin_number = static_cast<int>(std::floor(pixel/bin_range));
            histogram.at<float>(0, bin_number + bins)++;

            pixel = static_cast<float>(image.at<cv::Vec3b>(j, i)[2]);
            bin_number = static_cast<int>(std::floor(pixel/bin_range));
            histogram.at<float>(0, bin_number + bins + bins)++;

            // std::cout << bin_number << ", " << i << " "<< j  << "\n";
        }
    }
    if (is_norm) {
        cv::normalize(histogram, histogram, 0, 1, cv::NORM_MINMAX, -1);
    }
}


void ParticleFilterGPU::runObjectTracker(
    cv::Mat *img, cv::Rect &rect) {
    cv::Mat image = img->clone();
    if (image.empty()) {
       ROS_ERROR("NO IMAGE FRAME TO TRACK");
       return;
    }
    
    std::vector<Particle> x_particle = this->transition(
       this->particles_, this->dynamics, this->random_num_);

    /*
    this->printParticles(image, x_particle);
    cv::namedWindow("Tracking", cv::WINDOW_NORMAL);
    cv::imshow("Tracking", image);
    return;
    */
    
    std::clock_t start;
    double duration;
    start = std::clock();
    
    PFFeatures features;
    this->createParticlesFeature(features, image, x_particle);

    duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
    std::cout<<"Feature: "<< duration <<'\t';
    start = std::clock();
    
    std::vector<double> color_probability = this->colorHistogramLikelihood(
        x_particle, image, features, this->reference_features_);

    duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
    std::cout<<"Likelihood: "<< duration <<'\n';


    
    std::vector<double> wN;
    for (int i = 0; i < NUM_PARTICLES; i++) {
      double probability = static_cast<double>(
          color_probability[i]);
        wN.push_back(probability);
    }
    std::vector<double> nWeights = this->normalizeWeight(wN);
    this->reSampling(this->particles_, x_particle, nWeights);
    
    this->printParticles(image, particles_);    
    Particle x_e = this->meanArr(this->particles_);
    cv::Rect b_rect = cv::Rect(
       x_e.x - rect.width/2, x_e.y - rect.height/2,
       this->width_, this->height_);
    cv::circle(image, cv::Point2f(x_e.x, x_e.y), 3,
               cv::Scalar(255, 0, 0), CV_FILLED);
    // cv::rectangle(image, b_rect, cv::Scalar(255, 0, 255), 2);
    rect = b_rect;
    cv::resize(image, image, cv::Size(
                  image.cols * downsize_, image.rows * downsize_));
    cv::namedWindow("Tracking", cv::WINDOW_NORMAL);
    cv::imshow("Tracking", image);
}

template<typename T>
T ParticleFilterGPU::EuclideanDistance(
    Particle a, Particle b, bool is_square) {
    T dist = std::pow((a.x - b.x), 2) + std::pow((a.y - b.y), 2);
    if (is_square) {
        dist = std::sqrt(dist);
    }
    return dist;
}


std::vector<double> ParticleFilterGPU::colorHistogramLikelihood(
    const std::vector<Particle> &particles, cv::Mat &image,
    const PFFeatures features, const PFFeatures prev_features) {
    if (features.color_hist.cols != prev_features.color_hist.cols ||
        particles.empty()) {
        return std::vector<double>();
    }

    // cv::Mat results;
    // this->intensityCorrelation(results, image, this->prev_frame_);
    // cv::imshow("results", results);
    
    
    std::vector<double> probability(static_cast<int>(features.color_hist.rows));
    double *p = &probability[0];
// #ifdef _OPENMP
// #pragma omp parallel for num_threads(threads_)
// #endif
    for (int i = 0; i < features.color_hist.rows; i++) {
        cv::Mat p_color = features.color_hist.row(i);
        cv::Mat p_hog = features.hog_hist.row(i);
        double c_dist = DBL_MAX;
        double h_dist = DBL_MAX;
        int match_idx = -1;
        
        for (int j = 0; j < prev_features.color_hist.rows; j++) {
            // cv::Mat chist = prev_features.color_hist.row(j);
            // double d_color = cv::compareHist(
            //     chist, p_color, CV_COMP_BHATTACHARYYA);
            // if (d_color < c_dist) {
            //     c_dist = d_color;
            //     match_idx = j;
            // }
            
            cv::Mat hhist = prev_features.hog_hist.row(j);
            double d_hog = cv::compareHist(hhist, p_hog, CV_COMP_BHATTACHARYYA);
            double pt_dist = this->EuclideanDistance<double>(
                this->particles_[j], particles[i]);
            if (d_hog < h_dist) {
                h_dist = d_hog;
                match_idx = j;
            }
        }

        double prob = 0.0;
        if (match_idx != -1) {
            // h_dist = cv::compareHist(prev_features.hog_hist.row(match_idx),
            //                          p_hog, CV_COMP_BHATTACHARYYA);
            c_dist = cv::compareHist(prev_features.color_hist.row(match_idx),
                                     p_color, CV_COMP_BHATTACHARYYA);
            double c_prob = 1 * exp(-0.70 * c_dist);
            double h_prob = 1 * exp(-0.70 * h_dist);
            prob = c_prob * h_prob;
            double val = 0.0;
            if (prob < 0.7) {
                prob = 0.0;
            } else if (prob > 0.9) {
                this->reference_features_.color_hist.row(match_idx) =
                    features.color_hist.row(i);
                this->reference_features_.hog_hist.row(match_idx) =
                    features.hog_hist.row(i);
            } else if (prob > 0.7 && prob < 0.9) {
                const float adapt = prob;
                cv::Mat color_ref = reference_features_.color_hist.row(match_idx);
                cv::Mat hog_ref = reference_features_.hog_hist.row(match_idx);
                for (int y = 0; y < color_ref.cols; y++) {
                    color_ref.at<float>(0, y) *= (adapt);
                    color_ref.at<float>(0, y) += (
                        (1.0f - adapt) * features.color_hist.row(i).at<float>(0, y));
                }
                for (int y = 0; y < hog_ref.cols; y++) {
                    hog_ref.at<float>(0, y) *= (adapt);
                    hog_ref.at<float>(0, y) += (
                        (1.0f- adapt) * features.hog_hist.row(i).at<float>(0, y));
                }
                this->reference_features_.color_hist.row(match_idx) = color_ref;
                this->reference_features_.hog_hist.row(match_idx) = hog_ref;
            }
            // std::cout << "Prob: " << p[i] << " " << val << " " << p[i]
            // * val  << "\n";
        }
        p[i] = prob;
    }
    cv::namedWindow("Ref", cv::WINDOW_NORMAL);
    cv::imshow("Ref", reference_features_.color_hist);

    return probability;
}



std::vector<double> ParticleFilterGPU::colorHistogramLikelihood(
    std::vector<cv::Mat> &obj_patch) {
    std::vector<double> prob_hist(NUM_PARTICLES);
    double *ph_ptr = &prob_hist[0];
    int i;
#ifdef _OPENMP
#pragma omp parallel for private(i)
#endif
    for (i = 0; i < NUM_PARTICLES; i++) {
       double dist = static_cast<double>(
          this->computeHistogramDistances(
             obj_patch[i], &this->reference_object_histogram_));
       double bg_dist = static_cast<double>(
          this->computeHistogramDistances(
             obj_patch[i], &this->reference_background_histogram_));
        double pr = 0.0;
        if (dist < bg_dist) {
            pr = 2 * exp(-2 * dist);
        }
        ph_ptr[i] = pr;
    }
    return prob_hist;
}

std::vector<cv::Mat> ParticleFilterGPU::particleHistogram(
    cv::Mat &image, std::vector<Particle> &p) {
    if (image.empty()) {
       return std::vector<cv::Mat>();
    }
    cv::Mat col_hist[NUM_PARTICLES];
    cv::Mat hog_hist[NUM_PARTICLES];
    int i;
#ifdef _OPENMP
#pragma omp parallel for private(i) shared(p, col_hist)
#endif
    for (i = 0; i < NUM_PARTICLES; i++) {
       cv::Rect p_rect = cv::Rect(p[i].x - block_size_/2,
                                  p[i].y - block_size_/2,
                                  block_size_,
                                  block_size_);
       roiCondition(p_rect, image.size());
       cv::Mat roi = image(p_rect).clone();
       cv::Mat h_D;       

       this->computeHistogram(roi, h_D, true);  // color histogram
       h_D = h_D.reshape(1,1);
       col_hist[i] = h_D;       
    }
    std::vector<cv::Mat> obj_histogram;
    for (i = 0; i < NUM_PARTICLES; i++) {
      obj_histogram.push_back(col_hist[i]);
    }
    return obj_histogram;
}

std::vector<cv::Mat> ParticleFilterGPU::imagePatchHistogram(
    cv::Mat &image) {
    if (image.empty()) {
       return std::vector<cv::Mat>();
    }
    const int OVERLAP = 2;
    std::vector<cv::Mat> patch_hist; 
    for (int j = 0; j < image.rows; j += (block_size_/OVERLAP)) {
       for (int i = 0; i < image.cols; i += (block_size_/OVERLAP)) {
           cv::Rect rect = cv::Rect(i, j, block_size_, block_size_);
           roiCondition(rect, image.size());
           cv::Mat roi = image(rect);
           cv::Mat h_MD;
           this->computeHistogram(roi, h_MD, true);
           h_MD = h_MD.reshape(1,1);
           
           // cv::Mat hog = this->computeHOG(roi);
           
           // cv::Mat features;
           // cv::hconcat(h_MD, hog, features);
           patch_hist.push_back(h_MD);
           // patch_hist.push_back(features);
        }
    }
    return patch_hist;
}

void ParticleFilterGPU::roiCondition(
    cv::Rect &rect, cv::Size imageSize) {
    if (rect.x < 0) {
        rect.x = 0;
        // rect.width = block_size_;
    }
    if (rect.y < 0) {
        rect.y = 0;
        // rect.height = block_size_;
    }
    if ((rect.height + rect.y) > imageSize.height) {
        rect.y -= ((rect.height + rect.y) - imageSize.height);
        // rect.height = block_size_;
    }
    if ((rect.width + rect.x) > imageSize.width) {
        rect.x -= ((rect.width + rect.x) - imageSize.width);
        // rect.width = block_size_;
    }
}




void ParticleFilterGPU::intensityCorrelation(
    cv::Mat &results, cv::Mat &image, const cv::Mat &templ_img) {
    int py_side = 2;
    int level = 2;
    cv::Size default_size = image.size();
    int scale = 2;
    cv::cuda::GpuMat d_results;
    cv::Size result_size;

    cv::cuda::GpuMat d_templ(templ_img);
    cv::cuda::GpuMat d_image(image);
    do {
        cv::cuda::GpuMat d_result;
        cv::Ptr<cv::cuda::TemplateMatching> match =
            cv::cuda::createTemplateMatching(image.type(),
                                             CV_TM_CCOEFF_NORMED);
        match->match(d_image, d_templ, d_result);
        cv::cuda::normalize(d_result, d_result, 0, 1, cv::NORM_MINMAX, -1);
        if (py_side < level) {
            cv::cuda::resize(d_result, d_result, result_size);
            cv::cuda::multiply(d_results, d_result, d_results);
        } else {
            d_results = d_result;
            result_size = d_result.size();
        }
        cv::cuda::pyrDown(d_image, d_image);
        cv::cuda::pyrDown(d_templ, d_templ);
    } while (py_side-- != 0);
    cv::cuda::normalize(d_results, d_results, 0, 1, cv::NORM_MINMAX, -1);
    d_results.download(results);
}

void ParticleFilterGPU::hogCorrelation(
    cv::Mat &results, cv::Mat &image, const cv::Mat &templ_img) {
    int py_side = 1;
    int level = 1;
    cv::Size default_size = image.size();
    int scale = 2;
    cv::cuda::GpuMat d_results;
    cv::Size result_size;

    cv::cuda::GpuMat d_templ(templ_img);
    cv::cuda::GpuMat d_image(image);

    const int stride = 10;
    const cv::Size w_size = cv::Size(64/2, 64/2);
    cv::cuda::resize(d_templ, d_templ, w_size);

    cv::cuda::cvtColor(d_image, d_image, CV_BGR2GRAY);
    cv::cuda::cvtColor(d_templ, d_templ, CV_BGR2GRAY);

    cv::Ptr<cv::cuda::HOG> hog = cv::cuda::HOG::create(w_size);
    cv::cuda::GpuMat d_templ_descriptor;
    hog->compute(d_templ, d_templ_descriptor);
    cv::Mat templ_descriptor;
    d_templ_descriptor.download(templ_descriptor);
    
    results = cv::Mat::zeros(image.size(), CV_32FC1);
    
    do{
        cv::cuda::GpuMat d_result;
        cv::Rect rect;
        for (int j = stride/2; j < image.rows - stride; j += stride/2) {
            for (int i = stride/2; i < image.cols - stride; i += stride/2) {
                rect = cv::Rect(i - stride/2, j - stride/2,
                                w_size.width, w_size.height);

                if (rect.x + rect.width < image.cols &&
                    rect.y + rect.height < image.rows) {
                    cv::cuda::GpuMat d_descriptor;
                    hog->compute(d_image(rect), d_descriptor);

                    cv::Mat descriptor;
                    d_descriptor.download(descriptor);
                    double d = cv::compareHist(descriptor, templ_descriptor,
                                               CV_COMP_CHISQR);
                
                    // cv::Mat temp_result;
                    // if (!d_result.empty) {
                    //     d_result.download(temp_result);
                    // }
                    for (int y = j; y < j + w_size.height; y++) {
                        for (int x = i; x < i + w_size.width; x++) {
                            results.at<float>(y, x) += d;
                        }
                    }
                }
                
                // d_result.upload(temp_result);
            }
        }
        
        // if (py_side < level) {
        //     cv::cuda::resize(d_result, d_result, result_size);
        //     cv::cuda::multiply(d_results, d_result, d_results);
        // } else {
        //     d_results = d_result;
        //     result_size = d_result.size();
        // }
        //cv::cuda::pyrDown(d_image, d_image);
        //cv::cuda::pyrDown(d_templ, d_templ);
    } while(py_side-- != 0);
    // d_results.download(results);

    cv::normalize(results, results, 0, 1, cv::NORM_MINMAX, -1);
    std::cout << results  << "\n";
}

    


float l2Distance(cv::Vec3b a, cv::Vec3b b) {
    float i = (static_cast<float>(a[0]) - static_cast<float>(b[0]));
    float j = (static_cast<float>(a[1]) - static_cast<float>(b[1]));
    float k = (static_cast<float>(a[2]) - static_cast<float>(b[2]));
    float dist = (i * i) + (j * j) + (k * k);
    if (isnan(dist)) {
        return 0.0;
    }
    return sqrt(dist);
}

void ParticleFilterGPU::multiResolutionColorContrast(
    cv::Mat &image, const cv::Mat &prev_frame, const int p_layers) {
    if (image.empty() || prev_frame.empty() || p_layers < 1) {
        ROS_ERROR("ERROR CANNOT COMPUTER COLOR RESOLTION");
        return;
    }

    ROS_INFO("CONSTRACT");
    
    cv::Size default_size = image.size();
    cv::Mat prev_image = prev_frame.clone();
    // cv::cvtColor(image, image, CV_BGR2Lab);
    // cv::cvtColor(prev_image, prev_image, CV_BGR2Lab);
    // cv::resize(image, image, cv::Size(image.cols/2, image.rows/2));
        
    int size = 0;
    int padding = 3;
    
    cv::Mat prob = cv::Mat::zeros(image.size(), CV_32FC1);
    do {
#pragma omp parallel for num_threads(8) collapse(2)
        for (int i = padding; i < image.rows - padding; i++) {
            for (int j = padding; j < image.cols - padding; j++) {
                cv::Vec3b pixel = image.at<cv::Vec3b>(i, j);
                float distance = 0.0;
                int icounter = 0;
                /*
                for (int y = 0; y < prev_image.rows; y++) {
                    for (int x = 0; x < prev_image.cols; x++) {
                        cv::Vec3b p_pixel = prev_image.at<cv::Vec3b>(y, x);
                        distance += l2Distance(pixel, p_pixel);
                        icounter++;
                    }
                }
                if (icounter == 0) {
                    distance = 0.0f;
                } else {
                    distance /= static_cast<float>(icounter);
                }
                prob.at<float>(i, j) += distance;
                */
                for (int y = -padding; y <= padding ; y++) {
                    for (int x = -padding; x <= padding; x++) {
                        cv::Vec3b p_pixel = image.at<cv::Vec3b>(y, x);
                        distance += l2Distance(pixel, p_pixel);
                        icounter++;
                    }
                }
                if (icounter == 0) {
                    distance = 0.0f;
                } else {
                    distance /= static_cast<float>(icounter);
                }
                prob.at<float>(i, j) += distance;
            }
        }
        cv::pyrDown(image, image, cv::Size(image.cols/2, image.rows/2));
        cv::pyrDown(prob, prob, cv::Size(prob.cols/2, prob.rows/2));
        cv::pyrDown(prev_image, prev_image,
                    cv::Size(prev_image.cols/2, prev_image.rows/2));
        
    } while (size++ < p_layers);


        // std::cout << prob << "\n";
    cv::normalize(prob, prob, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());
    
    cv::resize(image, image, default_size);
    cv::resize(prob, prob, default_size);
    cv::namedWindow("prob", cv::WINDOW_NORMAL);
    cv::imshow("prob", prob);
}



/**
 * test
 
 */
void ParticleFilterGPU::superPixel(cv::Mat &image) {
    if (image.empty()) {
        return;
    }
    cv::Mat temp = image.clone();
    /*
    cv::Ptr<cv::ximgproc::SuperpixelSEEDS> seeds;
    bool is_init = false;
    int num_superpixels = 800;
    bool double_step = false;
    int num_levels = 1;
    int prior = 2;
    int num_hist_bins = 10;
    int num_iter = 10;

    
    seeds = cv::ximgproc::createSuperpixelSEEDS(
        image.cols, image.rows, image.channels(), num_superpixels,
        num_levels, prior, num_hist_bins, double_step);

    cv::Mat c_img;
    cv::cvtColor(image, c_img, CV_BGR2HSV);
    seeds->iterate(c_img, num_iter);

    cv::Mat labels;
    seeds->getLabels(labels);

    cv::Mat mask;
    seeds->getLabelContourMask(mask, false);

    cv::Mat results = image.clone();
    results.setTo(cv::Scalar(0, 255, 0), mask);

    // std::cout << labels << "\n";

    cv::namedWindow("results", cv::WINDOW_NORMAL);
    cv::imshow("results", results);
    cv::imshow("labels", mask);
    */
    
    // make sift keypoints
    cv::Ptr<cv::Feature2D> sift = cv::xfeatures2d::SIFT::create(0, 5, 0.05);

    cv::Mat templ_img = cv::imread("/home/krishneel/Desktop/25m.jpg");
    std::vector<cv::KeyPoint> keypoints_tmp;
    sift->detect(templ_img, keypoints_tmp);
    
    std::vector<cv::KeyPoint> keypoints;
    sift->detect(temp, keypoints);

    cv::Mat desc_tmp, desc;
    sift->compute(temp, keypoints, desc);
    sift->compute(templ_img, keypoints_tmp, desc_tmp);

    cv::BFMatcher matcher;
    std::vector<cv::DMatch> matches;
    matcher.match(desc_tmp, desc, matches);
    
    cv::Mat img_match;
    cv::drawMatches(templ_img, keypoints_tmp, temp, keypoints, matches,
                    img_match, cv::Scalar(0, 255, 0));
    
    
    cv::drawKeypoints(temp, keypoints, temp);
    cv::namedWindow("sift", cv::WINDOW_NORMAL);
    cv::imshow("sift", img_match);
    
}

int main(int argc, char *argv[]) {
    ros::init(argc, argv, "gpu_particle_filter");
    ParticleFilterGPU pfg;
    ros::spin();
    return 0;
}
