
// #include <gpu_particle_filter/gpu_particle_filter.h>
#include <gpu_particle_filter/particle_filter_kernel.h>

#define CUDA_ERROR_CHECK(process) {                \
        cudaAssert((process), __FILE__, __LINE__); \
    }                                              \

void cudaAssert(cudaError_t code, char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %dn",
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

__host__ __device__
struct cuImage{
    unsigned char pixel[3];
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
    // int *color_hist; //[COLOR_BINS * COLOR_CHANNEL];
    // int *hog_hist; //[36];
    int color_hist[COLOR_BINS * COLOR_CHANNEL];
    int hog_hist[36];
};


// __host__
cuParticles particles_[PARTICLES_SIZE];

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

__device__ __forceinline__
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
            transition[i] = sum + cuGenerateGaussian(global_state, offset) * G_SIGMA;
        }        
        cuParticles nxt_particle;
        nxt_particle.x = transition[0];
        nxt_particle.y = transition[1];
        nxt_particle.dx = transition[2];
        nxt_particle.dy = transition[3];
        prop_particles[offset] = nxt_particle;
    }
}

__global__
void cuPFPropagation(
    cuParticles *trans_particles,
    cuParticles *particles, curandState_t *global_state) {
    cuPFTransition(trans_particles, particles, global_state);    
}

__host__
void cuInitParticles(
    cuParticles *particles, float *box_corners,
    const dim3 block_size, const dim3 grid_size) {

    srand(time(NULL));
    
    size_t dim = 4 * sizeof(float);
    float *d_corners;
    cudaMalloc((void**)&d_corners, dim);
    cudaMemcpy(d_corners, box_corners, dim, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_state_, PARTICLES_SIZE * sizeof(curandState_t));
    curandInit<<<grid_size, block_size>>>(d_state_, unsigned(time(NULL)));

    cuParticles *d_particles;
    cudaMalloc((void**)&d_particles, sizeof(cuParticles) * PARTICLES_SIZE);
    cuPFInitalizeParticles<<<grid_size, block_size>>>(
        d_particles, d_state_, d_corners);
    dim = PARTICLES_SIZE * sizeof(cuParticles);
    cudaMemcpy(particles, d_particles, dim, cudaMemcpyDeviceToHost);

    // particles_ = particles;
    
    cudaFree(d_corners);
    cudaFree(d_particles);    
}


/**
 * features on GPU
 */
__global__ __forceinline__
void cuPFColorHistogram(
    int *histogram, cuImage *image, int width, int height) {
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
    cuImage *image, cuParticles particle, int width, int height, int patch_sz) {
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
    cuPFFeature *features, cuImage *image, cuParticles *particles,
    int width, int height, int patch_sz) {
    int t_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int t_idy = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = t_idx + t_idy * blockDim.x * gridDim.x;

    int *histo;
    if (offset < PARTICLES_SIZE) {
        // features[offset].color_hist = cuPFGetROIHistogram(
        //     image, particles[offset], width, height, patch_sz);
        
        histo = cuPFGetROIHistogram(
            image, particles[offset], width, height, patch_sz);
        
        for (int i = 0; i < COLOR_BINS * COLOR_CHANNEL; i++) {
            features[offset].color_hist[i] = 0;
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
                        features[offset].color_hist[bin_num + (z * COLOR_BINS)]++;
                    }
                }
            }
        }
    }
    
    __syncthreads();
    
    for (int i = 0; i < COLOR_BINS * COLOR_CHANNEL; i++) {
        printf("%d  %d \n", features[offset].color_hist[i], histo[i]);
    }
    printf("\n------------------------------------\n");
    
}


void gpuHist(cv::Mat image, cv::Mat cpu_hist) {
    const int SIZE = static_cast<int>(image.rows * image.cols) * sizeof(cuImage);
    cuImage *pixels = (cuImage*)malloc(sizeof(cuImage) * SIZE);
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

    
    cuImage *d_image;
    cudaMalloc((void**)&d_image, SIZE);
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
            std::cout << "DIFF: " << cpu_hist.at<float>(j ,i)  << "  "
                      << histogram[offset]  << "\n";
        }
    }
    std::cout << cpu_hist.size() <<  "\n"  << "\n";
}






/**
 * 
 */
cuPFFeatures ref_features_;
std::vector<Particle> cpu_particles_;

__host__
void particleFilterGPU(cv::Mat &image, cv::Rect &rect, bool &is_init) {
    const int SIZE = static_cast<int>(image.rows * image.cols) * sizeof(cuImage);
    cuImage *pixels = (cuImage*)malloc(sizeof(cuImage) * SIZE);
    for (int j = 0; j < image.rows; j++) {
        for (int i = 0; i < image.cols; i++) {
            int index = i + (j * image.cols);
            for (int k = 0; k < 3; k++) {
                pixels[index].pixel[k] = static_cast<unsigned char>(
                    image.at<cv::Vec3b>(j, i)[k]);
            }
        }
    }
    
    float *box_corners = (float*)malloc(4 * sizeof(float));
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


    bool is_on = true;
    if (is_init) {
        printf("\033[34m INITIALIZING TRACKER \033[0m\n");
        cuParticles particles[PARTICLES_SIZE];
        cuInitParticles(particles, box_corners, block_size, grid_size);

        for (int i = 0; i < PARTICLES_SIZE; i++) {
            particles_[i] = particles[i];
        }
        
        // compute weight -----------------------
        for(int i = 0; i < PARTICLES_SIZE; i++) {
            Particle p;
            p.x = (double)particles_[i].x;
            p.y = particles_[i].y;
            p.dx = particles_[i].dx;
            p.dy = particles_[i].dy;
            cpu_particles_.push_back(p);
        }
        cuCreateParticlesFeature(ref_features_, image, cpu_particles_, 1);
        printf("\033[34m TRACKER INITIALIZED  \033[0m\n");
        /// ---------------------------------------        
        
        is_init = false;
    } else if (is_on) {

        printf("\033[32m PROPAGATION  \033[0m\n");
        
        cuParticles *d_particles;
        cudaMalloc((void**)&d_particles, sizeof(cuParticles) * PARTICLES_SIZE);
        cudaMemcpy(d_particles, particles_, sizeof(cuParticles) * PARTICLES_SIZE,
                   cudaMemcpyHostToDevice);   // fix to keep previous
                                              // particles
        
        cuParticles *d_trans_particles;
        cudaMalloc((void**)&d_trans_particles,
                   sizeof(cuParticles) * PARTICLES_SIZE);
        cuPFPropagation<<<grid_size, block_size>>>(d_trans_particles,
                                                   d_particles, d_state_);
        cuParticles *x_particles = (cuParticles*)malloc(
            sizeof(cuParticles) * PARTICLES_SIZE);
        cudaMemcpy(x_particles, d_trans_particles, sizeof(cuParticles) *
                   PARTICLES_SIZE, cudaMemcpyDeviceToHost);


        // ********************************
        // cudaDeviceSynchronize();
        cuPFFeature *d_features;
        cudaMalloc((void**)&d_features, sizeof(cuPFFeature) * PARTICLES_SIZE);

        cuImage *d_image;
        cudaMalloc((void**)&d_image, SIZE);
        cudaMemcpy(d_image, pixels, SIZE, cudaMemcpyHostToDevice);

        
        cuPFParticleFeatures<<<grid_size, block_size>>>(
            d_features, d_image, d_particles, image.cols, image.rows, 8);
        
        
        cuPFFeature *particles_features = (cuPFFeature*)malloc(
            sizeof(cuPFFeature) * PARTICLES_SIZE);
        cudaMemcpy(particles_features, d_features, sizeof(cuPFFeature) *
                   PARTICLES_SIZE, cudaMemcpyDeviceToHost);

        /*
        std::cout << "PRINTING....."  << "\n";

        for (int i = 0; i < PARTICLES_SIZE; i++) {
            std::cout << "PROCESSING: " << i  << "\n";
            for (int j = 0; j < COLOR_BINS * COLOR_CHANNEL; j++) {
                std::cout << particles_features[i].color_hist[j] << ", "
                          << j << "  -- ";
            }
            std::cout << "\n";
        }        
        */
        return;
        //*******************************/
        

        
        printf("\033[32m COMPUTING WEIGHT  \033[0m\n");
        
        // compute weight -----------------------
        std::vector<Particle> cpu_particles;
        for(int i = 0; i < PARTICLES_SIZE; i++) {
            Particle p;
            p.x = (double)x_particles[i].x;
            p.y = x_particles[i].y;
            p.dx = x_particles[i].dx;
            p.dy = x_particles[i].dy;
            cpu_particles.push_back(p);
        }
        cuPFFeatures features;
        cuCreateParticlesFeature(features, image, cpu_particles, 1);
        std::vector<float> prob = cuHistogramLikelihood(
            cpu_particles, cpu_particles_, image, features, ref_features_);

        float *probabilities = (float*)malloc(sizeof(float) * PARTICLES_SIZE);
        for (int i = 0; i < prob.size(); i++) {
            probabilities[i] = prob[i];
        }
        float *d_probs;
        cudaMalloc((void**)&d_probs, sizeof(float) * PARTICLES_SIZE);
        cudaMemcpy(d_probs, probabilities, sizeof(float) * PARTICLES_SIZE,
                   cudaMemcpyHostToDevice);
        
        
        printf("\033[32m NORMALIZATION  \033[0m\n");
        
        cuPFNormalizeWeights<<<grid_size, block_size>>>(d_probs);
        cudaDeviceSynchronize();

        printf("\033[32m CUMULATIVE SUM  \033[0m\n");
        
        cuPFCumulativeSum<<<grid_size, block_size>>>(d_probs);
        cudaDeviceSynchronize();

        printf("\033[32m SAMPLING  \033[0m\n");
        
        cuPFSequentialResample<<<grid_size, block_size>>>(
            d_particles, d_trans_particles, d_probs, d_state_);

        cuParticles *update_part = (cuParticles*)malloc(sizeof(cuParticles) *
                                                        PARTICLES_SIZE);
        cudaMemcpy(update_part, d_particles, sizeof(cuParticles) * PARTICLES_SIZE,
                   cudaMemcpyDeviceToHost);

        printf("\033[32m UPDATING  \033[0m\n");
        
        for (int i = 0; i < PARTICLES_SIZE; i++) {
            particles_[i] = update_part[i];
        }
        
        printf("\033[31m DONE  \033[0m\n");

        /// ---------------------------------------

        
        for (int i = 0; i < PARTICLES_SIZE; i++) {
            cv::Point2f center = cv::Point2f(x_particles[i].x,
                                             x_particles[i].y);
            cv::circle(image, center, 3, cv::Scalar(0, 255, 255), CV_FILLED);

            center = cv::Point2f(update_part[i].x, update_part[i].y);
            cv::circle(image, center, 3, cv::Scalar(0, 255, 0), CV_FILLED);
        }    
    }
    
    for (int i = 0; i < PARTICLES_SIZE; i++) {
        cv::Point2f center = cv::Point2f(particles_[i].x,
                                         particles_[i].y);
        // cv::circle(image, center, 3, cv::Scalar(255, 0, 255), CV_FILLED);
    }
    
    /* normalize check */
    /*
    float weights[PARTICLES_SIZE] = {0.1, 0.2, 0.3, 0.4, 0.5,
                                     0.1, 0.2, 0.3, 0.4, 0.5,
                                     0.1, 0.2, 0.3, 0.4, 0.5,
                                     0.1, 0.2, 0.3, 0.4, 0.5,
                                     0.1, 0.2, 0.3, 0.4, 0.5,
                                     0.1, 0.2, 0.3, 0.4, 0.5,
                                     0.1, 0.2, 0.3, 0.4, 0.5,
                                     0.1, 0.2, 0.3, 0.4, 0.5,
                                     0.1, 0.2, 0.3, 0.4, 0.5,
                                     0.1, 0.2, 0.3, 0.4, 0.5
    };
    float *d_weight;
    cudaMalloc((void**)&d_weight, sizeof(float) * PARTICLES_SIZE);
    cudaMemcpy(d_weight, weights, sizeof(float) * PARTICLES_SIZE,
               cudaMemcpyHostToDevice);

    cuPFNormalizeWeights<<<grid_size, block_size>>>(d_weight);
    
    float *nweights = (float*)malloc(sizeof(float) * PARTICLES_SIZE);
    cudaMemcpy(nweights, d_weight, sizeof(float) * PARTICLES_SIZE,
               cudaMemcpyDeviceToHost);
    float sum = 0.0;
    for (int i = 0; i < PARTICLES_SIZE; i++) {
        std::cout << nweights[i]  << ", ";
        sum += nweights[i];
    }
    std::cout  << "\n Sum: " << sum << "\n\n";

    float *cweights = (float*)malloc(sizeof(float) * PARTICLES_SIZE);
    cuPFCumulativeSum<<<grid_size, block_size>>>(d_weight);
    cudaMemcpy(cweights, d_weight, sizeof(float) * PARTICLES_SIZE,
               cudaMemcpyDeviceToHost);
    sum = 0.0;
    for (int i = 0; i < PARTICLES_SIZE; i++) {
        std::cout << cweights[i]  << ", ";
        sum += cweights[i];
    }
    std::cout  << "\n Sum: " << sum << "\n";
    /*-----------------*/

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
    // free(particles);
    // cudaFree(d_state_);
}
