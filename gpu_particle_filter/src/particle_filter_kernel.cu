
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

__device__ __forceinline__
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

__device__ __forceinline__
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

__device__ __forceinline__
void cuPFSequentialResample(
    cuParticles *particles, cuParticles *prop_particles, float *weights) {
    cuPFNormalizeWeights(weights);
    __shared__ cuParticles cache[PARTICLES_SIZE];
    int t_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int t_idy = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = t_idx + t_idy * blockDim.x * gridDim.x;
    if (offset < PARTICLES_SIZE) {
        cache[offset] = particles[offset];
    }
    __syncthreads();

    if (offset == 0) {
        const float s_ptx = 1.0f/PARTICLES_SIZE; // change to gaussian
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
    
    if (offset < PARTICLES_SIZE) {
        particles[offset] = cache[offset];
    }
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
            transition[i] = sum + cuGenerateGaussian(global_state, offset);;
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

    cudaFree(d_corners);
    cudaFree(d_particles);    
}

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
    
    if (is_init) {
        printf("\033[34m INITIALIZING TRACKER \033[0m\n");
        cuInitParticles(particles_, box_corners, block_size, grid_size);

        // compute weight
        
        is_init = false;
    } else {
        cuParticles *d_particles;
        cudaMalloc((void**)&d_particles, sizeof(cuParticles) * PARTICLES_SIZE);
        cudaMemcpy(d_particles, particles_, sizeof(cuParticles) * PARTICLES_SIZE,
                   cudaMemcpyHostToDevice);   // fix to keep previous particles
        cuParticles *d_trans_particles;
        cudaMalloc((void**)&d_trans_particles,
                   sizeof(cuParticles) * PARTICLES_SIZE);
        cuPFPropagation<<<grid_size, block_size>>>(d_trans_particles,
                                                   d_particles, d_state_);
        cuParticles *x_particles = (cuParticles*)malloc(
            sizeof(cuParticles) * PARTICLES_SIZE);
        cudaMemcpy(x_particles, d_trans_particles, sizeof(cuParticles) *
                   PARTICLES_SIZE, cudaMemcpyDeviceToHost);

        // compute weight
        
        
        for (int i = 0; i < PARTICLES_SIZE; i++) {
            cv::Point2f center = cv::Point2f(x_particles[i].x,
                                             x_particles[i].y);
            cv::circle(image, center, 3, cv::Scalar(0, 255, 255), CV_FILLED);
        }    
    }
    
    for (int i = 0; i < PARTICLES_SIZE; i++) {
        cv::Point2f center = cv::Point2f(particles_[i].x,
                                         particles_[i].y);
        cv::circle(image, center, 3, cv::Scalar(255, 0, 255), CV_FILLED);
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

    // cuPFNormalizeWeights<<<grid_size, block_size>>>(d_weight);
    // cuPFCumulativeSum<<<grid_size, block_size>>>(d_weight);

    
    float *nweights = (float*)malloc(sizeof(float) * PARTICLES_SIZE);
    cudaMemcpy(nweights, d_weight, sizeof(float) * PARTICLES_SIZE,
               cudaMemcpyDeviceToHost);
    float sum = 0.0;
    for (int i = 0; i < PARTICLES_SIZE; i++) {
        std::cout << nweights[i]  << ", ";
        sum += nweights[i];
    }
    std::cout  << "\n Sum: " << sum << "\n";
    /*-----------------*/

    cudaEventRecord(d_stop, 0);
    cudaEventSynchronize(d_stop);
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, d_start, d_stop);
    std::cout << "\033[33m ELAPSED TIME:  \033[0m" << elapsed_time/1000.0f
              << "\n";

    cudaEventDestroy(d_start);
    cudaEventDestroy(d_stop);
    
    free(box_corners);
    // free(particles);
    // cudaFree(d_state_);
}
