
#include <gpu_particle_filter/color_histogram_kernel.h>

void getHistogram(cv::Mat &histogram, const cv::Mat &image,
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

void roiCondition(cv::Rect &rect, cv::Size imageSize) {
    if (rect.x < 0) {
        rect.x = 0;
    }
    if (rect.y < 0) {
        rect.y = 0;
    }
    if ((rect.height + rect.y) > imageSize.height) {
        rect.y -= ((rect.height + rect.y) - imageSize.height);
    }
    if ((rect.width + rect.x) > imageSize.width) {
        rect.x -= ((rect.width + rect.x) - imageSize.width);
    }
}


bool cuCreateParticlesFeature(cuPFFeatures &features, const cv::Mat &img,
                            const std::vector<Particle> &particles,
                            const int downsize) {
    if (img.empty() || particles.empty()) {
        return false;
    }
    
    cv::Size wsize = cv::Size(16/downsize, 16/downsize);
    cv::Size bsize = cv::Size(16/downsize, 16/downsize);
    cv::Size csize = cv::Size(8/downsize, 8/downsize);
    cv::Ptr<cv::cuda::HOG> hog = cv::cuda::HOG::create(wsize, bsize, csize);
    
    const int LENGHT = static_cast<int>(particles.size());
    const int dim = 8/downsize;
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
        
        roiCondition(rect, image.size());
        cv::Mat roi = image(rect).clone();
        cv::Mat part_hist;
        getHistogram(part_hist, roi, bin_size, 3, false);

        cv::Point2i s_pt = cv::Point2i((particles[i].x - dim),
                                       (particles[i].y - dim));        
        
        for (int j = 0; j < 2; j++) {
            for (int k = 0; k < 2; k++) {
                cv::Rect_<int> s_rect = cv::Rect_<int>(s_pt.x, s_pt.y, dim, dim);
                roiCondition(s_rect, image.size());
                roi = image(s_rect).clone();
                cv::Mat hist;
                getHistogram(hist, roi, bin_size, 3, false);

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
        roiCondition(rect, image.size());
        roi = image(rect).clone();
        cv::Mat region_hist;
        getHistogram(region_hist, roi, bin_size, 3, false);
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
        hog->compute(d_roi, d_desc);
        cv::Mat desc;
        d_desc.download(desc);
        hog_descriptors.push_back(desc);

        color_hist.push_back(histogram[i]);
    }
    
    features.hog_hist = hog_descriptors;
    features.color_hist = color_hist;
    return true;
}



float EuclideanDistance(const Particle a, const Particle b, bool is_square) {
    float dist = std::pow((a.x - b.x), 2) + std::pow((a.y - b.y), 2);
    if (is_square) {
        dist = std::sqrt(dist);
    }
    return dist;
}


std::vector<float> cuHistogramLikelihood(
    const std::vector<Particle> &particles,
    const std::vector<Particle> &particles_, 
    cv::Mat &image, const cuPFFeatures features, const cuPFFeatures prev_features) {
    if (features.color_hist.cols != prev_features.color_hist.cols ||
        particles.empty()) {
        return std::vector<float>();
    }    
    std::vector<float> probability(static_cast<int>(features.color_hist.rows));
    float *p = &probability[0];
    for (int i = 0; i < features.color_hist.rows; i++) {
        cv::Mat p_color = features.color_hist.row(i);
        cv::Mat p_hog = features.hog_hist.row(i);
        float c_dist = DBL_MAX;
        float h_dist = DBL_MAX;
        int match_idx = -1;
        
        for (int j = 0; j < prev_features.color_hist.rows; j++) {
            
            cv::Mat hhist = prev_features.hog_hist.row(j);
            float d_hog = cv::compareHist(hhist, p_hog, CV_COMP_BHATTACHARYYA);
            float pt_dist = EuclideanDistance(particles_[j], particles[i], true);
            if (d_hog < h_dist) {
                h_dist = d_hog;
                match_idx = j;
            }
        }
        float prob = 0.0;
        if (match_idx != -1) {
            c_dist = cv::compareHist(prev_features.color_hist.row(match_idx),
                                     p_color, CV_COMP_BHATTACHARYYA);
            float c_prob = 1 * exp(-0.70 * c_dist);
            float h_prob = 1 * exp(-0.70 * h_dist);
            prob = c_prob * h_prob;
            float val = 0.0;
            if (prob < 0.7) {
                prob = 0.0;
            } else if (prob > 0.9) {
                prev_features.color_hist.row(match_idx) =
                    features.color_hist.row(i);
                prev_features.hog_hist.row(match_idx) =
                    features.hog_hist.row(i);
            } else if (prob > 0.7 && prob < 0.9) {
                /*
                const float adapt = prob;
                cv::Mat color_ref = prev_features.color_hist.row(match_idx);
                cv::Mat hog_ref = prev_features.hog_hist.row(match_idx);
                for (int y = 0; y < color_ref.cols; y++) {
                    color_ref.at<float>(0, y) *= (adapt);
                    color_ref.at<float>(0, y) += (
                        (1.0f - adapt) * features.color_hist.row(
                        i).at<float>(0, y));
                }
                for (int y = 0; y < hog_ref.cols; y++) {
                    hog_ref.at<float>(0, y) *= (adapt);
                    hog_ref.at<float>(0, y) += (
                        (1.0f- adapt) * features.hog_hist.row(i).at<float>(0, y));
                }
                prev_features.color_hist.row(match_idx) = color_ref;
                prev_features.hog_hist.row(match_idx) = hog_ref;
                */
            }

        }
        p[i] = prob;
    }

    return probability;
}
