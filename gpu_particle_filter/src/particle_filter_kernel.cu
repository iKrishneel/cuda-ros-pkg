
#include <gpu_particle_filter/particle_filter_kernel.h>

#define CUDA_ERROR_CHECK(process) {                \
        cudaAssert((process), __FILE__, __LINE__); \
    }                                              \

void cudaAssert(cudaError_t code, char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
       fprintf(stderr, "GPUassert: %s %s %dn",
               cudaGetErrorString(code), file, line);
       if (abort) {
          exit(code);
       }
    }
}

typedef struct __align__(16) {
    int width;
    int height;
    int stride;
    float *elements;
} cuMat;

__host__ __device__
struct cuParticles{
    float x;
    float y;
    float dx;
    float dy;
};


template<class T, int N> struct __align__(16) cuImage{
    // unsigned char pixel[N];
    T pixel[N];
};

__device__
struct cuRect {
    int x;
    int y;
    int width;
    int height;
};

__host__ __device__
struct cuPFFeature {
    // int *color_hist;
    // int *hog_hist;
    float color_hist[COLOR_BINS * COLOR_CHANNEL];
    float hog_hist[36];
};


// __host__
cuParticles particles_[PARTICLES_SIZE];
cuPFFeature particles_features_[PARTICLES_SIZE];
curandState_t *d_state_;

/**
 * Cuda Shared memory operations
 */
__device__
float getElement(const cuMat A, int row, int col) {
    return A.elements[row * A.stride + col];
}

__device__
void setElement(cuMat A, int row, int col, float value) {
    A.elements[row + A.stride + col] = value;
}

__device__
cuMat getSubMatrix(cuMat A, int row, int col) {
    cuMat a_sub;
    a_sub.width = BLOCK_SIZE;
    a_sub.height = BLOCK_SIZE;
    a_sub.stride = A.stride;
    a_sub.elements = &A.elements[A.stride * BLOCK_SIZE * row
                                 + BLOCK_SIZE * col];
    return a_sub;
}

__device__
cuMat cuMatrixProduct(cuMat A, cuMat B, cuMat c) {
    int block_row = blockIdx.y;
    int block_col = blockIdx.x;
    cuMat c_sub = getSubMatrix(c, block_row, block_col);
    float c_value = 0.0f;

    int row = threadIdx.y;
    int col = threadIdx.x;

    for (int i = 0; i < (A.width / BLOCK_SIZE); i++) {
        cuMat a_sub = getSubMatrix(A, block_row, i);
        cuMat b_sub = getSubMatrix(B, i, block_col);

        __shared__ float as[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float bs[BLOCK_SIZE][BLOCK_SIZE];

        as[row][col] = getElement(a_sub, row, col);
        bs[row][col] = getElement(b_sub, row, col);
        __syncthreads();

        for (int j = 0; j < BLOCK_SIZE; j++) {
            c_value += as[row][j] * bs[j][col];
        }
        __syncthreads();
        // setElement(c, row, col, c_value);
        setElement(c_sub, row, col, c_value);
        // c.elements[row * c.width + col] = c_value;
    }
}

/**
 * Tracking
 */

__device__ __constant__
float DYNAMICS[STATE_SIZE][STATE_SIZE] = {{1, 0, 1, 0},
                                          {0, 1, 0, 1},
                                          {0, 0, 1, 0},
                                          {0, 0, 0, 1}};

__device__
cuMat getPFDynamics() {
    int t_idx = threadIdx.x + blockIdx.x * blockDim.x;
    cuMat dynamics;
    if (t_idx == 0) {
        dynamics.width = STATE_SIZE;
        dynamics.height = STATE_SIZE;
        dynamics.stride = STATE_SIZE;
        float cu_dyna[STATE_SIZE * STATE_SIZE] = {1, 0, 1, 0,
                                                  0, 1, 0, 1,
                                                  0, 0, 1, 0,
                                                  0, 0, 0, 1};
        dynamics.elements = cu_dyna;
    }
    __syncthreads();
    return dynamics;
}


__global__
void curandInit(
    curandState_t *state, unsigned long seed) {
    int t_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int t_idy = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = t_idx + t_idy * blockDim.x * gridDim.x;
    curand_init(seed, offset, 0, &state[offset]);
}

__device__ __forceinline__
float cuGenerateUniform(
    curandState_t *global_state, int idx) {
    curandState_t local_state = global_state[idx];
    float rnd_num = curand_uniform(&local_state);
    global_state[idx] = local_state;
    return rnd_num;
}

__device__ __forceinline__
float cuGenerateGaussian(
    curandState_t *global_state, int idx) {
    curandState_t local_state = global_state[idx];
    float rnd_num = curand_normal(&local_state);
    global_state[idx] = local_state;
    return rnd_num;
}

__global__ __forceinline__
void cuPFInitalizeParticles(
    cuParticles *particles, curandState_t *global_state, float *box_corners) {
    int t_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int t_idy = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = t_idx + t_idy * blockDim.x * gridDim.x;

    // printf(" BLOCK: %d\n", blockDim.x);
    
    // int offset = t_idx;
    if (offset < PARTICLES_SIZE) {
        cuParticles particle;
        particle.x = cuGenerateUniform(global_state, offset);
        particle.y = cuGenerateUniform(global_state, offset);
        particle.dx = cuGenerateUniform(global_state, offset);
        particle.dy = cuGenerateUniform(global_state, offset);

        particle.x *= box_corners[2] - box_corners[0] + 0.999999f;
        particle.x += box_corners[0];
        particle.y *= box_corners[3] - box_corners[1] + 0.999999f;
        particle.y += box_corners[1];

        const float rate = 2.0f;
        particle.dx *= (rate - 0.0f + 0.999999f);
        particle.dx += 0.0f;
        particle.dy *= (rate - 0.0f + 0.999999f);
        particle.dy += 0.0f;

        particles[offset] = particle;

        // printf("offset: %d, %f, %f\n", offset, particle.x, particle.y);
    }
}

__host__ __device__ __align__(16)
int cuDivUp(int a, int b) {
    return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

__global__ __forceinline__
void cuPFNormalizeWeights(float *weights) {
    __shared__ float cache[PARTICLES_SIZE];
    int t_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int cache_index = threadIdx.x;
    float temp = 0.0f;
    
    while (t_idx < PARTICLES_SIZE) {
        temp += weights[t_idx];
        t_idx += blockDim.x * gridDim.x;
    }
    
    cache[cache_index] = temp;
    __syncthreads();
    
    int i = blockDim.x/2;
    while (i != 0) {
        if (cache_index < i) {
            cache[cache_index] += cache[cache_index + i];
        }
        __syncthreads();
        i /= 2;
    }
    float sum = 0.0f;
    if (cache_index == 0) {
        sum = cache[0];
    }
    __syncthreads();

    t_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int t_idy = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = t_idx + t_idy * blockDim.x * gridDim.x;

    if (offset < PARTICLES_SIZE && sum != 0.0f) {
        weights[offset] /= sum;
    }
}

__global__ __forceinline__
void cuPFCumulativeSum(float *weights) {
    __shared__ float weight_cache[PARTICLES_SIZE];
    int t_idx = threadIdx.x + blockIdx.x * blockDim.x;
    weight_cache[t_idx] = weights[t_idx];
    
    int cache_index = threadIdx.x;
    while (t_idx < PARTICLES_SIZE) {
        if (t_idx > 0) {
            weight_cache[t_idx] = weights[t_idx] + weight_cache[t_idx - 1];
        }
        __syncthreads();
        t_idx += blockDim.x * gridDim.x;
    }
    __syncthreads();
    
    t_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int t_idy = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = t_idx + t_idy * blockDim.x * gridDim.x;

    if (offset < PARTICLES_SIZE) {
        weights[offset] = weight_cache[offset];
    }
}

// __device__ __forceinline__
__global__
void cuPFSequentialResample(
    cuParticles *particles, cuParticles *prop_particles,
    float *weights, curandState_t *global_state) {

    __shared__ cuParticles cache[PARTICLES_SIZE];
    int t_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int t_idy = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = t_idx + t_idy * blockDim.x * gridDim.x;

    if (offset < PARTICLES_SIZE) {
        cache[offset] = particles[offset];
    }
    __syncthreads();

    if (offset == 0) {
        // const float s_ptx = 1.0f/PARTICLES_SIZE; // change to gaussian
        float s_ptx = abs(cuGenerateGaussian(global_state, offset));
        s_ptx *= (1.0f / PARTICLES_SIZE);
        int cdf_stx = 1;
        for (int i = 0; i < PARTICLES_SIZE; i++) {
            float ptx = s_ptx + (1.0/PARTICLES_SIZE) * (i - 1);
            while (ptx > weights[cdf_stx]) {
                cdf_stx++;
            }
            cache[i] = prop_particles[cdf_stx];
        }
    }

    __syncthreads();

    // t_idx = threadIdx.x + blockIdx.x * blockDim.x;
    // t_idy = threadIdx.y + blockIdx.y * blockDim.y;
    // offset = t_idx + t_idy * blockDim.x * gridDim.x;
    if (offset < PARTICLES_SIZE) {
        particles[offset] = cache[offset];
    }
    __syncthreads();
}

__device__
void cuPFCenterRms(
    cuParticles *center, const cuParticles *particles) {
    int t_idx = threadIdx.x + blockIdx.x * blockDim.x;
    cuParticles sum;
    sum.x = 0.0f;
    sum.y = 0.0f;
    sum.dx = 0.0f;
    sum.dy = 0.0f;
    while (t_idx < PARTICLES_SIZE) {
        sum.x += particles[t_idx].x;
        sum.y += particles[t_idx].y;
        sum.dx += particles[t_idx].dx;
        sum.dy += particles[t_idx].dy;
        t_idx += blockDim.x * gridDim.x;
    }
    __syncthreads();

    center->x = static_cast<float>(sum.x/PARTICLES_SIZE);
    center->y = static_cast<float>(sum.y/PARTICLES_SIZE);
    center->dx = static_cast<float>(sum.dx/PARTICLES_SIZE);
    center->dy = static_cast<float>(sum.dy/PARTICLES_SIZE);
}

__global__ __forceinline__
void cuPFTransition(cuParticles *prop_particles,
    const cuParticles *particles, curandState_t *global_state) {
    int t_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int t_idy = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = t_idx + t_idy * blockDim.x * gridDim.x;
    
    if (offset < PARTICLES_SIZE) {
        float element[STATE_SIZE] = {particles[offset].x,
                                     particles[offset].y,
                                     particles[offset].dx,
                                     particles[offset].dy};
        float transition[STATE_SIZE];
        for (int i = 0; i < STATE_SIZE; i++) {
            float sum = 0.0f;
            for (int j = 0; j < STATE_SIZE; j++) {
                sum += DYNAMICS[i][j] * element[j];
            }
            transition[i] = sum + cuGenerateGaussian(
               global_state, offset) * G_SIGMA;
        }
        cuParticles nxt_particle;
        nxt_particle.x = transition[0];
        nxt_particle.y = transition[1];
        nxt_particle.dx = transition[2];
        nxt_particle.dy = transition[3];
        prop_particles[offset] = nxt_particle;
    }
}

/**
 * FIX TO COMPUTE 3.0
 */
__global__ __forceinline__
void cuPFPropagation(
    cuParticles *trans_particles,
    cuParticles *particles, curandState_t *global_state) {
    // cuPFTransition(trans_particles, particles, global_state);
}



/**
 * HOG
 */

__device__ __forceinline__
void cuPFHOGBinVoting(float *angle, int *lower_index) {
    float nearest_lower = 1e6;
    *lower_index = 0;
    for (int i = HOG_BIN_ANGLE/2; i < HOG_ANGLE; i += HOG_BIN_ANGLE) {
       float distance = fabs(*angle - i);
       if (static_cast<float>(i) < *angle) {
          if (distance < nearest_lower) {
             nearest_lower = distance;
             *lower_index = i;
          }
       }
    }
}

__device__ __forceinline__
void cuPFHOGBlockGradient(cuImage<float, HOG_FEATURE_DIMS> *block_hog,
                          cuImage<float, HOG_NBINS> *bins,
                          const int index, const int stride) {
    int icounter = 0;
    float ssums = 0.0f;
    for (int j = 0; j < HOG_BLOCK; j++) {
       for (int i = 0; i < HOG_BLOCK; i++) {
          int ind = i + (j + stride) + index;
          for (int k = 0; k < HOG_NBINS; k++) {
             block_hog[0].pixel[icounter] = bins[ind].pixel[k];
             ssums += block_hog[0].pixel[icounter];
             icounter++;
          }
       }
    }
    ssums = sqrtf(ssums);
    for (int i = 0; i < HOG_FEATURE_DIMS; i++) {
       block_hog[0].pixel[i] /= ssums;
    }
}

__global__ __forceinline__
void cuPFHOGImageGradient(cuImage<float, HOG_NBINS> *im_grad,
                          cuImage<float, 1> *im_ang, int width, int height) {
    int  t_idx = (threadIdx.x + blockIdx.x * blockDim.x);
    int t_idy = (threadIdx.y + blockIdx.y * blockDim.y);
    int offset = t_idx + t_idy * blockDim.x * gridDim.x;
    
    int image_size = (width * height)/(HOG_CELL * HOG_CELL);

    extern __shared__ cuImage<float, HOG_NBINS> orientation_histogram[];
    __shared__ int icounter;
    if (offset == 0) {
       icounter = 0;
    }
    __syncthreads();

    if (offset < image_size) {
       int index = offset/HOG_CELL;
       // for (int i = 0; i < HOG_NBINS; i++) {
       //    orientation_histogram[offset].pixel[i] = 0.0f;
       // }
       atomicAdd(&icounter, 1);
    }
    __syncthreads();

    if (offset < image_size) {
       float bin[HOG_NBINS];
       for (int i = 0; i < HOG_NBINS; i++) {
          bin[i] = 0;
       }
       int index = offset * HOG_CELL;
       for (int j = t_idy; j < t_idy + HOG_CELL; j++) {
          for (int i = t_idx; i < t_idx + HOG_CELL; i++) {
             float angle = static_cast<float>(im_ang[index].pixel[0]);
             int l_bin = 0;
             cuPFHOGBinVoting(&angle, &l_bin);
             float l_ratio = 1.0f - (angle - l_bin)/HOG_BIN_ANGLE;
             int l_index = (l_bin - (HOG_BIN_ANGLE/2))/HOG_BIN_ANGLE;
             bin[l_index] += (im_ang[index].pixel[0] * l_ratio);
             bin[l_index + 1] += (im_ang[index].pixel[0] * (1.0f - l_ratio));
          }
       }
       for (int i = 0; i < HOG_NBINS; i++) {
          orientation_histogram[offset].pixel[i] = bin[i];
       }
    }

    __shared__ int stride;
    if (offset == 0) {
       stride = static_cast<int>(width/HOG_CELL);
    }
    __syncthreads();
    
    if (offset < image_size) {
       int index = offset * HOG_CELL;
       cuImage<float, HOG_FEATURE_DIMS> block_hog[1];
       cuPFHOGBlockGradient(block_hog, orientation_histogram, index, stride);

       
    }

}

/**
 * END HOG
 */





/**
 * features on GPU
 */
__global__ __forceinline__
void cuPFColorHistogram(
    int *histogram, cuImage<unsigned char, 3> *image, int width, int height) {
    int t_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int t_idy = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = t_idx + t_idy * blockDim.x * gridDim.x;

    int bin_range = static_cast<int>(ceilf(256/COLOR_BINS));
    if (offset < (height * width)) {
        int b = static_cast<int>(image[offset].pixel[0]);
        int bin_num_b = static_cast<int>(floorf(b/bin_range));
        atomicAdd(&histogram[bin_num_b], 1);
        
        int g = static_cast<int>(image[offset].pixel[1]);
        int bin_num_g = static_cast<int>(floorf(g/bin_range));
        atomicAdd(&histogram[bin_num_g + COLOR_BINS], 1);
        
        int r = static_cast<int>(image[offset].pixel[2]);
        int bin_num_r = static_cast<int>(floorf(r/bin_range));
        atomicAdd(&histogram[bin_num_r + (2 * COLOR_BINS)], 1);
    }
    __syncthreads();
}

__device__ __forceinline__
void cuPFRoiCondition(
    cuRect *rect, int width, int height) {
    if (rect->x < 0) {
        rect->x = 0;
    }
    if (rect->y < 0) {
        rect->y = 0;
    }
    if ((rect->width + rect->x) > width) {
        rect->x -= ((rect->width + rect->x) - width);
    }
    if ((rect->height + rect->y) > height) {
        rect->y -= ((rect->height + rect->y) - height);
    }
}


__device__ __forceinline__
int *cuPFGetROIHistogram(
    cuImage<unsigned char, 3> *image, cuParticles particle, int width,
    int height, int patch_sz) {
    int histogram[COLOR_BINS * COLOR_CHANNEL];
    for (int i = 0; i < COLOR_BINS * COLOR_CHANNEL; i++) {
        histogram[i] = 0;
    }
    if (patch_sz < 2 || width < patch_sz || height < patch_sz) {
        return histogram;
    }
    
    cuRect rect;
    rect.x = particle.x - (patch_sz/2);
    rect.y = particle.y - (patch_sz/2);
    rect.width = rect.x + patch_sz;
    rect.height = rect.y + patch_sz;
    cuPFRoiCondition(&rect, width, height);
    
    int bin_range = static_cast<int>(ceilf(256/COLOR_BINS));
    for (int y = rect.y; y < rect.height; y++) {
        for (int x = rect.x; x < rect.width; x++) {
            int offset = x + (y * width);
            if (offset < (height * width)) {
                for (int z = 0; z < COLOR_CHANNEL; z++) {
                    int b = static_cast<int>(image[offset].pixel[z]);
                    int bin_num = static_cast<int>(floorf(b/bin_range));
                    histogram[bin_num + (z * COLOR_BINS)]++;
                }
            }
        }
    }
    
    return histogram;
}

__global__ __forceinline__
void cuPFParticleFeatures(
    cuPFFeature *color_features, cuImage<unsigned char, 3> *image, cuParticles *particles,
    int width, int height, int patch_sz) {
    int t_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int t_idy = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = t_idx + t_idy * blockDim.x * gridDim.x;

    // __shared__ int max_values[PARTICLES_SIZE];
    cuPFFeature features;
    
    int max_values;
    if (offset < PARTICLES_SIZE) {
        // features[offset].color_hist = cuPFGetROIHistogram(
        //     image, particles[offset], width, height, patch_sz);

        // max_values[offset] = 0;
        
        for (int i = 0; i < COLOR_BINS * COLOR_CHANNEL; i++) {
            features.color_hist[i] = 0.0;
        }
        cuRect rect;
        rect.x = particles[offset].x - (patch_sz/2);
        rect.y = particles[offset].y - (patch_sz/2);
        rect.width = rect.x + patch_sz;
        rect.height = rect.y + patch_sz;
        cuPFRoiCondition(&rect, width, height);

        int bin_range = static_cast<int>(ceilf(256/COLOR_BINS));
        for (int y = rect.y; y < rect.height; y++) {
            for (int x = rect.x; x < rect.width; x++) {
                int index = x + (y * width);
                if (index < (height * width)) {
                    for (int z = 0; z < COLOR_CHANNEL; z++) {
                        int b = static_cast<int>(image[index].pixel[z]);
                        int bin_num = static_cast<int>(floorf(b/bin_range));
                        features.color_hist[bin_num + (z * COLOR_BINS)]+= 1.0f;
                        // max_values[offset]++;  // sum of features
                        max_values++;
                    }
                }
            }
        }
    }
    
    // __syncthreads();
    
    // normalize
    if (offset < PARTICLES_SIZE) {
        // float max_val = static_cast<float>(max_values[offset]);
        for (int i = 0; i < COLOR_BINS * COLOR_CHANNEL; i++) {
            color_features[offset].color_hist[i] =
                features.color_hist[i] / static_cast<float>(max_values);
        }
    }
}

/**
 * HOG features on gpu via OpenCV
 */
__host__
void computeHOG(cuPFFeature *features, cuParticles *particles,
                cv::Mat image, const int patch_sz, const int downsize) {
    cv::Size wsize = cv::Size(patch_sz/downsize, patch_sz/downsize);
    cv::Size bsize = cv::Size(patch_sz/downsize, patch_sz/downsize);
    cv::Size csize = cv::Size(patch_sz/(downsize * 2), patch_sz/(downsize * 2));
    /*
    cv::Ptr<cv::cuda::HOG> hog = cv::cuda::HOG::create(wsize, bsize, csize);
    for (int i = 0; i < PARTICLES_SIZE; i++) {
        cv::Rect_<int> rect = cv::Rect_<int>(particles[i].x - patch_sz/2,
                                             particles[i].y - patch_sz/2,
                                             patch_sz, patch_sz);
        if (rect.x < 0) {
            rect.x = 0;
        }
        if (rect.y < 0) {
            rect.y = 0;
        }
        if ((rect.height + rect.y) > image.rows) {
            rect.y -= ((rect.height + rect.y) - image.rows);
        }
        if ((rect.width + rect.x) > image.cols) {
            rect.x -= ((rect.width + rect.x) - image.cols);
        }
        cv::cuda::GpuMat d_roi(image(rect));
        cv::cuda::GpuMat d_desc;
        cv::cuda::cvtColor(d_roi, d_roi, CV_BGR2GRAY);
        hog->compute(d_roi, d_desc);
        cv::Mat desc;
        d_desc.download(desc);
        
        for (int j = 0; j < desc.rows; j++) {
            for (int k = 0; k < desc.cols; k++) {
                features[i].hog_hist[k + (j * patch_sz)] =
                    static_cast<float>(desc.at<float>(j, k));
            }
        }
    }
    */
}


/**
 * color histogram TEST FUNCTION
 */

void gpuHist(cv::Mat image, cv::Mat cpu_hist) {
    const int SIZE = static_cast<int>(image.rows *
                                      image.cols) * sizeof(cuImage<unsigned char, 3>);
    cuImage<unsigned char, 3> *pixels = reinterpret_cast<cuImage<unsigned char, 3>*>(
       malloc(sizeof(cuImage<unsigned char, 3>) * SIZE));
    for (int j = 0; j < image.rows; j++) {
        for (int i = 0; i < image.cols; i++) {
            int index = i + (j * image.cols);
            for (int k = 0; k < 3; k++) {
                pixels[index].pixel[k] = static_cast<unsigned char>(
                    image.at<cv::Vec3b>(j, i)[k]);
            }
        }
    }

    cudaEvent_t d_start;
    cudaEvent_t d_stop;
    cudaEventCreate(&d_start);
    cudaEventCreate(&d_stop);
    cudaEventRecord(d_start, 0);

    
    cuImage<unsigned char, 3> *d_image;
    cudaMalloc(reinterpret_cast<void**>(&d_image), SIZE);
    cudaMemcpy(d_image, pixels, SIZE, cudaMemcpyHostToDevice);

    
    dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_size(cuDivUp(image.cols, BLOCK_SIZE),
                   cuDivUp(image.rows, BLOCK_SIZE));

    int *d_histogram;
    size_t MEM_SIZE = COLOR_BINS * COLOR_CHANNEL * sizeof(int);
    cudaMalloc((void**)&d_histogram, MEM_SIZE);
    cudaMemset(d_histogram, 0, MEM_SIZE);
    
    cuPFColorHistogram<<<grid_size, block_size>>>(d_histogram, d_image,
                                                  image.cols,
                                                  image.rows);
    
    int *histogram = (int*)malloc(MEM_SIZE);
    cudaMemcpy(histogram, d_histogram, MEM_SIZE, cudaMemcpyDeviceToHost);


    cudaEventRecord(d_stop, 0);
    cudaEventSynchronize(d_stop);
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, d_start, d_stop);
    std::cout << "\033[33m ELAPSED TIME:  \033[0m" << elapsed_time/1000.0f
              << "\n";
    
    for (int j = 0; j < cpu_hist.rows; j++) {
        for (int i = 0; i < cpu_hist.cols; i++) {
            int offset = i + (j * cpu_hist.cols);
            std::cout << "DIFF: " << cpu_hist.at<float>(j, i)  << "  "
                      << histogram[offset]  << "\n";
        }
    }
    std::cout << cpu_hist.size() <<  "\n"  << "\n";
}


/**
 * histogram likelihood
 */

__device__ __forceinline__
float cuEuclideanDist(cuParticles *a, cuParticles *b) {
    float dist = ((a->x - b->x) * (a->x - b->x)) +
       ((a->y - b->y) * (a->y - b->y));
    return sqrtf(dist);
}


__device__ __forceinline__
float cuBhattacharyyaDist(float *histA, float *histB, int dimension) {
    float prod = 0.0f;
    float norm_a = 0.0f;
    float norm_b = 0.0f;
    for (int i = 0; i < dimension; i++) {
        prod += (sqrtf(histA[i] * histB[i]));
        norm_a += histA[i];
        norm_b += histB[i];
    }
    norm_a /= static_cast<float>(dimension);
    norm_b /= static_cast<float>(dimension);

    float norm = 1.0f / sqrtf(norm_a * norm_b * powf(dimension, 2));
    return sqrtf(1.0f - norm * prod);
}

__global__
void cuPFParticleLikelihoods(
    float *probs, cuParticles *particles, cuParticles *templ_particles,
    cuPFFeature *features, cuPFFeature *templ_features) {
    
    int t_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int t_idy = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = t_idx + t_idy * blockDim.x * gridDim.x;

    if (offset < PARTICLES_SIZE) {
        
        float h_dist = FLT_MAX;
        int match_idx = -1;
        for (int i = 0; i < PARTICLES_SIZE; i++) {
            // float d_hog = cuBhattacharyyaDist(
            //    templ_features[i].hog_hist,
            //    features[offset].hog_hist, HOG_DIM);

            float d_hog = cuBhattacharyyaDist(
                templ_features[i].color_hist,
                features[offset].color_hist, COLOR_BINS * COLOR_CHANNEL);

            if (d_hog < h_dist) {
                h_dist = d_hog;
                match_idx = i;
            }
        }

        // printf("%d  %f\n", offset, h_dist);
        
        float prob = 0.0f;
        if (match_idx != -1) {
            // float c_dist = cuBhattacharyyaDist(
            //     templ_features[match_idx].color_hist,
            //     features[match_idx].color_hist, COLOR_BINS * COLOR_CHANNEL);
            // prob = 1.0f * expf(-COLOR_CONTRL * c_dist) *
            //     1.0f * expf(-HOG_CONTRL * h_dist);
            float c_dist = h_dist;
            prob = 1.0f * expf(-COLOR_CONTRL * c_dist);
            
            if (prob < PROBABILITY_THRESH) {
                prob = 0.0f;
            }
        }
        probs[offset] = prob;
    }
}

__host__
void cuInitParticles(
    cuParticles *particles, float *box_corners, cv::Mat image,
    const dim3 block_size, const dim3 grid_size) {

    srand(time(NULL));
    
    size_t dim = 4 * sizeof(float);
    float *d_corners;
    // cudaMalloc((void**)&d_corners, dim);

    cudaMalloc(reinterpret_cast<void**>(&d_corners), dim);
    
    cudaMemcpy(d_corners, box_corners, dim, cudaMemcpyHostToDevice);
    cudaMalloc(reinterpret_cast<void**>(&d_state_),
               PARTICLES_SIZE * sizeof(curandState_t));
    curandInit<<<grid_size, block_size>>>(d_state_, unsigned(time(NULL)));

    cuParticles *d_particles;
    cudaMalloc(reinterpret_cast<void**>(&d_particles),
               sizeof(cuParticles) * PARTICLES_SIZE);
    cuPFInitalizeParticles<<<grid_size, block_size>>>(d_particles,
                                                      d_state_, d_corners);
    dim = PARTICLES_SIZE * sizeof(cuParticles);
    cudaMemcpy(particles, d_particles, dim, cudaMemcpyDeviceToHost);

    // cudaDeviceSynchronize();

    const int SIZE = static_cast<int>(image.rows * image.cols) *
       sizeof(cuImage<unsigned char, 3>);
    cuImage<unsigned char, 3> *pixels = reinterpret_cast<cuImage<unsigned char, 3>*>(
       malloc(sizeof(cuImage<unsigned char, 3>) * SIZE));
    for (int j = 0; j < image.rows; j++) {
        for (int i = 0; i < image.cols; i++) {
            int index = i + (j * image.cols);
            for (int k = 0; k < 3; k++) {
                pixels[index].pixel[k] = static_cast<unsigned char>(
                    image.at<cv::Vec3b>(j, i)[k]);
            }
        }
    }
    cuImage<unsigned char, 3> *d_image;
    cudaMalloc(reinterpret_cast<void**>(&d_image), SIZE);
    cudaMemcpy(d_image, pixels, SIZE, cudaMemcpyHostToDevice);
        
    cuPFFeature *d_features;  // open this if errror****
    cudaMalloc(reinterpret_cast<void**>(&d_features),
               sizeof(cuPFFeature) * PARTICLES_SIZE);
        
    cuPFParticleFeatures<<<grid_size, block_size>>>(
        d_features, d_image, d_particles, image.cols, image.rows, PATCH_SIZE/2);
    
    // cuPFFeature *particles_features = (cuPFFeature*)malloc(
    //     sizeof(cuPFFeature) * PARTICLES_SIZE);
    cudaMemcpy(particles_features_, d_features, sizeof(cuPFFeature) *
               PARTICLES_SIZE, cudaMemcpyDeviceToHost);

    // cudaDeviceSynchronize();

    // computeHOG(particles_features_, particles, image, PATCH_SIZE, 1);
    
    // cudaMemcpy(d_features, particles_features_, sizeof(cuPFFeature) *
    //            PARTICLES_SIZE, cudaMemcpyHostToDevice);

    cudaFree(d_image);
    cudaFree(d_corners);
    cudaFree(d_particles);
    cudaFree(d_features);
}



/**
 * 
 */
// cuPFFeatures ref_features_;
// std::vector<Particle> cpu_particles_;

__host__
void particleFilterGPU(cv::Mat &image, cv::Rect &rect, bool &is_init) {
    const int SIZE = static_cast<int>(image.rows * image.cols) *
       sizeof(cuImage<unsigned char, 3>);
    cuImage<unsigned char, 3> *pixels = reinterpret_cast<cuImage<
       unsigned char, 3>*>(malloc(sizeof(cuImage<unsigned char, 3>) * SIZE));
    for (int j = 0; j < image.rows; j++) {
        for (int i = 0; i < image.cols; i++) {
            int index = i + (j * image.cols);
            for (int k = 0; k < 3; k++) {
                pixels[index].pixel[k] = static_cast<unsigned char>(
                    image.at<cv::Vec3b>(j, i)[k]);
            }
        }
    }

    // float *box_corners = (float*)malloc(4 * sizeof(float));
    float *box_corners = reinterpret_cast<float*>(malloc(4 * sizeof(float)));
    box_corners[0] = rect.x;
    box_corners[1] = rect.y;
    box_corners[2] = rect.x + rect.width;
    box_corners[3] = rect.y + rect.height;
    
    dim3 block_size(1, PARTICLES_SIZE);
    dim3 grid_size(1, 1);

    cudaEvent_t d_start;
    cudaEvent_t d_stop;
    cudaEventCreate(&d_start);
    cudaEventCreate(&d_stop);
    cudaEventRecord(d_start, 0);

    
    /**
     * TEST
     */

    cuImage<unsigned char, 3> *d_image;
    cudaMalloc(reinterpret_cast<void**>(&d_image), SIZE);
    cudaMemcpy(d_image, pixels, SIZE, cudaMemcpyHostToDevice);
    
    cuImage<float, HOG_NBINS> *d_hog;
    cudaMalloc(reinterpret_cast<void**>(&d_hog),
               sizeof(cuImage<float, HOG_NBINS>) * image.cols * image.rows);

    cuImage<float, 1> *d_temp;
    cudaMalloc(reinterpret_cast<void**>(&d_temp),
               sizeof(cuImage<float, 1>) * image.cols * image.rows);

    cuPFHOGImageGradient<<<grid_size, block_size>>>(d_hog, d_temp,
                                                    640, 480);
    
    // cudaDeviceSynchronize();
    cudaEventRecord(d_stop, 0);
    cudaEventSynchronize(d_stop);
    float elapsed_time1;
    cudaEventElapsedTime(&elapsed_time1, d_start, d_stop);
    std::cout << "\033[33m ELAPSED TIME:  \033[0m" << elapsed_time1/1000.0f
              << "\n";
    
    cudaFree(d_hog);
    cudaFree(d_temp);
    cudaFree(d_image);
    free(pixels);
    return;
    /**
     * end test
     */



    

    bool is_on = true;
    if (is_init) {
        printf("\033[34m INITIALIZING TRACKER \033[0m\n");
        
        cuParticles particles[PARTICLES_SIZE];
        cuInitParticles(particles, box_corners, image, block_size, grid_size);
        
        for (int i = 0; i < PARTICLES_SIZE; i++) {
           particles_[i] = particles[i];
        }

        printf("\033[34m TRACKER INITIALIZED  \033[0m\n");

        
        is_init = false;
    } else if (is_on) {
       
       printf("\033[32m PROPAGATION  \033[0m\n");
       
       cuParticles *d_particles;
       cudaMalloc(reinterpret_cast<void**>(&d_particles),
                  sizeof(cuParticles) * PARTICLES_SIZE);
       cudaMemcpy(d_particles, particles_, sizeof(cuParticles) * PARTICLES_SIZE,
                  cudaMemcpyHostToDevice);   // fix to keep previous particles
        
       cuParticles *d_trans_particles;
       cudaMalloc(reinterpret_cast<void**>(&d_trans_particles),
                  sizeof(cuParticles) * PARTICLES_SIZE);
       cuPFTransition<<<grid_size, block_size>>>(d_trans_particles,
                                                 d_particles, d_state_);
       
       cuParticles *x_particles = reinterpret_cast<cuParticles*>(
          malloc(sizeof(cuParticles) * PARTICLES_SIZE));
       cudaMemcpy(x_particles, d_trans_particles, sizeof(cuParticles) *
                  PARTICLES_SIZE, cudaMemcpyDeviceToHost);


       printf("\033[32m COMPUTING WEIGHT  \033[0m\n");

       // ********************************
       cuImage<unsigned char, 3> *d_image;
       cudaMalloc(reinterpret_cast<void**>(&d_image), SIZE);
       cudaMemcpy(d_image, pixels, SIZE, cudaMemcpyHostToDevice);
       
       cuPFFeature *d_features;
       cudaMalloc(reinterpret_cast<void**>(&d_features),
                  sizeof(cuPFFeature) * PARTICLES_SIZE);
        
       cuPFParticleFeatures<<<grid_size, block_size>>>(
          d_features, d_image, d_particles, image.cols,
          image.rows, PATCH_SIZE/2);

       printf("\033[32m COPYING TEMPLATE  \033[0m\n");
        // copy template features to device ---
       cuPFFeature *d_templ_feat;
       cudaMalloc(reinterpret_cast<void**>(&d_templ_feat),
                  sizeof(cuPFFeature) * PARTICLES_SIZE);
       cudaMemcpy(d_templ_feat, particles_features_, sizeof(cuPFFeature) *
                  PARTICLES_SIZE, cudaMemcpyHostToDevice);
       

       // compute probabilities
       printf("\033[32m PROBABILITY  \033[0m\n");
       float *d_probabilities;
       cudaMalloc(reinterpret_cast<void**>(&d_probabilities),
                  sizeof(float) * PARTICLES_SIZE);
       cudaMemset(d_probabilities, 0, sizeof(float) * PARTICLES_SIZE);

       cuPFParticleLikelihoods<<<grid_size, block_size>>>(
            d_probabilities, d_trans_particles,
            d_particles, d_features, d_templ_feat);

       
       // float probability[PARTICLES_SIZE];
       // cudaMemcpy(probability, d_probabilities,
       //            sizeof(float) * PARTICLES_SIZE, cudaMemcpyDeviceToHost);
       
       printf("\033[32m NORMALIZATION  \033[0m\n");
        
       cuPFNormalizeWeights<<<grid_size, block_size>>>(d_probabilities);
       cudaDeviceSynchronize();
       
       printf("\033[32m CUMULATIVE SUM  \033[0m\n");
       
       cuPFCumulativeSum<<<grid_size, block_size>>>(d_probabilities);
       cudaDeviceSynchronize();
       
        printf("\033[32m SAMPLING  \033[0m\n");
        
        cuPFSequentialResample<<<grid_size, block_size>>>(
           d_particles, d_trans_particles, d_probabilities, d_state_);

        cuParticles *update_part = reinterpret_cast<cuParticles*>(
           malloc(sizeof(cuParticles) * PARTICLES_SIZE));
        cudaMemcpy(update_part, d_particles, sizeof(cuParticles) *
                   PARTICLES_SIZE, cudaMemcpyDeviceToHost);
        
        printf("\033[32m UPDATING  \033[0m\n");
        
       for (int i = 0; i < PARTICLES_SIZE; i++) {
          particles_[i] = update_part[i];
        }
       
       printf("\033[31m DONE  \033[0m\n");
       

       for (int i = 0; i < PARTICLES_SIZE; i++) {
          cv::Point2f center = cv::Point2f(x_particles[i].x,
                                           x_particles[i].y);
          cv::circle(image, center, 3, cv::Scalar(255, 0, 255), CV_FILLED);

          center = cv::Point2f(particles_[i].x,
                                           particles_[i].y);
          cv::circle(image, center, 3, cv::Scalar(0, 255, 0), CV_FILLED);
       }
        cudaFree(d_features);
        cudaFree(d_probabilities);
        cudaFree(d_image);
        cudaFree(d_templ_feat);
        cudaFree(d_trans_particles);
        cudaFree(d_particles);
    }
    /*
    for (int i = 0; i < PARTICLES_SIZE; i++) {
        cv::Point2f center = cv::Point2f(particles_[i].x,
                                         particles_[i].y);
        cv::circle(image, center, 3, cv::Scalar(255, 0, 255), CV_FILLED);
    }
    */

    cudaEventRecord(d_stop, 0);
    cudaEventSynchronize(d_stop);
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, d_start, d_stop);
    std::cout << "\033[33m ELAPSED TIME:  \033[0m" << elapsed_time/1000.0f
              << "\n";


    cv::namedWindow("particels", cv::WINDOW_NORMAL);
    cv::imshow("particels", image);
    cv::waitKey(3);
    
    cudaEventDestroy(d_start);
    cudaEventDestroy(d_stop);
    
    free(box_corners);
    free(pixels);
    // free(particles);
    // cudaFree(d_state_);
}
