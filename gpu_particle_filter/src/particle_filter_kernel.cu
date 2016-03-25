
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
            __syncthreads();
        }
        setElement(c_sub, row, col, c_value);
    }
}

/**
 * Tracking
 */

__device__ __constant__
float cu_dynamics[STATE_SIZE][STATE_SIZE] = {{1, 0, 1, 0},
                                             {0, 1, 0, 1},
                                             {0, 0, 1, 0},
                                             {0, 0, 0, 1}};    

__device__
cuMat getPFDynamics() {
    cuMat dynamics;
    dynamics.width = STATE_SIZE;
    dynamics.height = STATE_SIZE;
    dynamics.stride = STATE_SIZE;
    float cu_dyna[STATE_SIZE * STATE_SIZE] = {1, 0, 1, 0,
                                              0, 1, 0, 1,
                                              0, 0, 1, 0,
                                              0, 0, 0, 1};
    dynamics.elements = cu_dyna;
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

__host__
void cuInitParticles(
    curandState_t *d_state, cuParticles *particles, float *box_corners,
    const dim3 block_size, const dim3 grid_size) {

    srand(time(NULL));
    
    size_t dim = 4 * sizeof(float);
    float *d_corners;
    cudaMalloc((void**)&d_corners, dim);
    cudaMemcpy(d_corners, box_corners, dim, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_state, PARTICLES_SIZE * sizeof(curandState_t));
    curandInit<<<grid_size, block_size>>>(d_state, unsigned(time(NULL)));

    cuParticles *d_particles;
    cudaMalloc((void**)&d_particles, sizeof(cuParticles) * PARTICLES_SIZE);
    cuPFInitalizeParticles<<<grid_size, block_size>>>(
        d_particles, d_state, d_corners);
    dim = PARTICLES_SIZE * sizeof(cuParticles);
    cudaMemcpy(particles, d_particles, dim, cudaMemcpyDeviceToHost);

    cudaFree(d_corners);
    cudaFree(d_particles);
    
}

// __device__
__global__
void cuPFNormalizeWeights(float *weights) {
    __shared__ float cache[PARTICLES_SIZE];
    int t_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int cache_index = threadIdx.x;
    float temp = 0.0f;

    printf("START: %d\n", t_idx);
    
    while (t_idx < PARTICLES_SIZE) {
        temp += weights[t_idx];
        t_idx += blockDim.x * gridDim.x;
    }
    cache[cache_index] = temp;
    __syncthreads();

    // printf("Temp: %f\n", temp);
    
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

__device__
void cuPFTransition(cuParticles *particles, curandState_t *global_state) {
    cuMat dynamics = getPFDynamics(); // move this
    
    int t_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int t_idy = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = t_idx + t_idy * blockDim.x * gridDim.x;
    if (offset < PARTICLES_SIZE) {
        cuMat part_mat;
        part_mat.width = 1;
        part_mat.height = STATE_SIZE;
        part_mat.stride = 1;
        float element[STATE_SIZE] = {particles[offset].x,
                                     particles[offset].y,
                                     particles[offset].dx,
                                     particles[offset].dy};
        part_mat.elements = element;
        cuMat transition;
        cuMatrixProduct(dynamics, part_mat, transition);

        cuParticles nxt_particle;
        nxt_particle.x = transition.elements[0] +
            cuGenerateGaussian(global_state, offset);
        nxt_particle.y = transition.elements[1] +
            cuGenerateGaussian(global_state, offset);
        nxt_particle.dx = transition.elements[2] +
            cuGenerateGaussian(global_state, offset);
        nxt_particle.dy = transition.elements[3] +
            cuGenerateGaussian(global_state, offset);

        particles[offset] = nxt_particle;
    }
}

__global__
void cuPFTracking(cuParticles *particles) {
    int t_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int t_idy = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = t_idx + t_idy * blockDim.x * gridDim.x;
    if (offset < PARTICLES_SIZE) {
        cuParticles particle;
        
    }
}


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
    
    curandState_t *d_state;
    cuParticles *particles = (cuParticles*)malloc(
        sizeof(cuParticles) * PARTICLES_SIZE);


    cudaEvent_t d_start;
    cudaEvent_t d_stop;
    cudaEventCreate(&d_start); 
    cudaEventCreate(&d_stop);
    cudaEventRecord(d_start, 0);
    
    if (!is_init) {
        cuInitParticles(d_state, particles, box_corners, block_size, grid_size);
        // is_init = false;

        // plot the particles
        for (int i = 0; i < PARTICLES_SIZE; i++) {
            cv::Point2f center = cv::Point2f(particles[i].x, particles[i].y);
            cv::circle(image, center, 3, cv::Scalar(0, 255, 255), CV_FILLED);
        }
    }

    /* normalize check */
    float weights[PARTICLES_SIZE] = {0.1, 0.2, 0.3, 0.4, 0.5};
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
    free(particles);
    cudaFree(d_state);
}



/**
 * 
 */

__global__
void hello(char *a, int *b) {
    a[threadIdx.x] += b[threadIdx.x];

}

void test_cuda(char* a, int* b) {

    char *ad;
    int *bd;
    const int csize = N*sizeof(char);
    const int isize = N*sizeof(int);

    printf("%s", a);

    cudaMalloc((void**)&ad, csize);
    cudaMalloc((void**)&bd, isize);
    cudaMemcpy(ad, a, csize, cudaMemcpyHostToDevice);
    cudaMemcpy(bd, b, isize, cudaMemcpyHostToDevice);

    dim3 dimBlock( BLOCK_SIZE, 1 );
    dim3 dimGrid( 1, 1 );
    hello<<<dimGrid, dimBlock>>>(ad, bd);
    cudaMemcpy(a, ad, csize, cudaMemcpyDeviceToHost);
    cudaFree(ad);
    cudaFree(bd);
}

template<typename T>
__device__ __forceinline__
T cuFloor(const T x) {
    return static_cast<T>(std::floor(x));
}

__global__
void boxFilterGPU(int *pixels, const int fsize) {
    int t_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int t_idy = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = t_idx + t_idy * blockDim.x * gridDim.x;

    int val = 0;
    int icounter = 0;
    if ((t_idx > fsize && t_idx < 640 - fsize) &&
        (t_idy > fsize && t_idy < 480 - fsize)) {
        for (int i = -fsize; i < fsize + 1; i++) {
            for (int j = -fsize; j < fsize + 1; j++) {
                int idx = (t_idx + j) + (t_idy + i) * blockDim.x * gridDim.x;
                val += pixels[idx];
                icounter++;
            }
        }
        pixels[offset] = val/icounter;
    }
}

void boxFilter(cv::Mat &image, const int size) {
    // cv::cvtColor(image, image, CV_BGR2GRAY);
    int lenght = static_cast<int>(image.rows * image.cols) * sizeof(int);
    int *pixels = (int*)malloc(lenght);

#ifdef _OPENMP
#pragma omp parallel for num_threads(8) collapse(2) shared(pixels)
#endif
    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            int index = j + (i * image.cols);
            pixels[index] = (int)image.at<uchar>(i, j);
        }
    }
    
    cudaEvent_t d_start;
    cudaEvent_t d_stop;
    cudaEventCreate(&d_start); 
    cudaEventCreate(&d_stop);
    cudaEventRecord(d_start, 0);
    
    int *d_pixels;
    cudaMalloc((void**)&d_pixels, lenght);
    cudaMemcpy(d_pixels, pixels, lenght, cudaMemcpyHostToDevice);
    
    dim3 dim_thread(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dim_block(static_cast<int>(image.cols)/dim_thread.x,
                   static_cast<int>(image.rows)/dim_thread.y);

    int b_start = static_cast<int>((float)size/2.0f);

    std::cout << "val: " << b_start  << "\n";
    
    boxFilterGPU<<<dim_block, dim_thread>>>(d_pixels, b_start);

    int *download_pixels = (int*)malloc(lenght);
    cudaMemcpy(download_pixels, d_pixels, lenght, cudaMemcpyDeviceToHost);

    cudaEventRecord(d_stop, 0);
    cudaEventSynchronize(d_stop);
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, d_start, d_stop);

    std::cout << "\033[33m ELAPSED TIME:  \033[0m" << elapsed_time/1000.0f
              << "\n";
    
    int stride = image.cols;
    int j = 0;
    int k = 0;
    for (int i = 0; i < image.cols * image.rows; i++) {
        if (i == stride) {
            j++;
            k = 0;
            stride += image.cols;
        }
        image.at<uchar>(j, k++) = download_pixels[i];
    }
    
    cudaFree(d_pixels);
    free(pixels);
    cudaEventDestroy(d_start);
    cudaEventDestroy(d_stop);
}


/*******************************************************
 **using defined struct
******************************************************/



__global__
void boxFilterManGPU(cuImage* d_pixels, const int fsize) {
    int t_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int t_idy = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = t_idx + t_idy * blockDim.x * gridDim.x;

    int val[3];
    val[0] = val[1] = val[2] = 0;
    int icounter = 0;
    if ((t_idx > fsize && t_idx < 640 - fsize) &&
        (t_idy > fsize && t_idy < 480 - fsize)) {
        for (int i = -fsize; i < fsize + 1; i++) {
            for (int j = -fsize; j < fsize + 1; j++) {
                int idx = (t_idx + j) + (t_idy + i) * blockDim.x * gridDim.x;
                val[0] += d_pixels[idx].pixel[0];
                val[1] += d_pixels[idx].pixel[1];
                val[2] += d_pixels[idx].pixel[2];
                icounter++;
            }
        }
        d_pixels[offset].pixel[0] = val[0]/icounter;
        d_pixels[offset].pixel[1] = val[1]/icounter;
        d_pixels[offset].pixel[2] = val[2]/icounter;
    }
}

__host__
void boxFilterMan(cv::Mat &image, const int fsize) {
    const int SIZE = static_cast<int>(image.rows * image.cols) *
        sizeof(cuImage);
    cuImage *pixels = (cuImage*)malloc(sizeof(cuImage) * SIZE);
#ifdef _OPENMP
#pragma omp parallel for num_threads(8) collapse(2) shared(pixels)
#endif
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
    
    cuImage *d_pixels;
    cudaMalloc((void**)&d_pixels, SIZE);
    cudaMemcpy(d_pixels, pixels, SIZE, cudaMemcpyHostToDevice);

    dim3 dim_thread(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dim_block(static_cast<int>(image.cols)/dim_thread.x,
                   static_cast<int>(image.rows)/dim_thread.y);
    boxFilterManGPU<<<dim_block, dim_thread>>>(d_pixels, fsize);

    cuImage *download_pixels = (cuImage*)malloc(SIZE);
    cudaMemcpy(download_pixels, d_pixels, SIZE, cudaMemcpyDeviceToHost);

    cudaEventRecord(d_stop, 0);
    cudaEventSynchronize(d_stop);
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, d_start, d_stop);

    std::cout << "\033[33m ELAPSED TIME:  \033[0m" << elapsed_time/1000.0f
              << "\n";
    
    int stride = image.cols;
    int j = 0;
    int k = 0;
    for (int i = 0; i < image.cols * image.rows; i++) {
        if (i == stride) {
            j++;
            k = 0;
            stride += image.cols;
        }
        image.at<cv::Vec3b>(j, k)[0] = download_pixels[i].pixel[0];
        image.at<cv::Vec3b>(j, k)[1] = download_pixels[i].pixel[1];
        image.at<cv::Vec3b>(j, k)[2] = download_pixels[i].pixel[2];
        k++;
    }
    
    cudaFree(d_pixels);
    free(pixels);
    free(download_pixels);
    cudaEventDestroy(d_start);
    cudaEventDestroy(d_stop);
}



/*******************************************************
using built in
 *******************************************************/

__global__
void boxFilter2DGPU(cudaPitchedPtr d_pitched_ptr, int COLS, int ROWS, int D) {
    int t_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int t_idy = threadIdx.y + blockIdx.y * blockDim.y;

    char* d_ptr = static_cast<char*>(d_pitched_ptr.ptr);
    size_t pitch = d_pitched_ptr.pitch;
    size_t slice_pitch = pitch * ROWS;
    
    int* element  = (int*)(d_ptr + t_idy * pitch) + t_idx;
    element[0] = 255;
    // element[1] = 0;
    // element[2] = 255;
    
    // for (int i = 0; i < D; i++) {
    //     char* slice = d_ptr + i * slice_pitch;
    //     float* row = (float*)(slice + t_idy * 1);
    //     row[t_idx] = 255;
    // }
}

__host__
void boxFilter2D(cv::Mat &image, const int fsize) {
    const int ROWS = static_cast<int>(image.rows);
    const int COLS = static_cast<int>(image.cols);
    const int DEPTH = 3;
    int pixels[COLS][ROWS][DEPTH];
    
    for (int j = 0; j < ROWS; j++) {
        for (int i = 0; i < COLS; i++) {
            for (int k = 0; k < DEPTH; k++) {
                // pixels[i][j][k] = static_cast<int>(
                //     image.at<cv::Vec3b>(j, i)[k]);
                pixels[i][j][k] = 0;
            }
        }
    }
    cudaExtent extent = make_cudaExtent(COLS * sizeof(int), ROWS, DEPTH);
    cudaPitchedPtr d_pitched_ptr;
    cudaMalloc3D(&d_pitched_ptr, extent);
    
    cudaMemcpy3DParms d_parms = {0};
    d_parms.srcPtr.ptr = pixels;
    d_parms.srcPtr.pitch = COLS * sizeof(int);
    d_parms.srcPtr.xsize = COLS;
    d_parms.srcPtr.ysize = ROWS;
    
    d_parms.dstPtr.ptr = d_pitched_ptr.ptr;
    d_parms.dstPtr.pitch = d_pitched_ptr.pitch;
    d_parms.dstPtr.xsize = COLS;
    d_parms.dstPtr.ysize = ROWS;

    d_parms.extent.width = COLS * sizeof(int);
    d_parms.extent.height = ROWS;
    d_parms.extent.depth = DEPTH;
    d_parms.kind = cudaMemcpyHostToDevice;
    
    cudaMemcpy3D(&d_parms);
    
    dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_size(cuDivUp(COLS, BLOCK_SIZE), cuDivUp(ROWS, BLOCK_SIZE));

    // std::cout << "PITCH: " << d_pitched_ptr.pitch  << "\n";
    
    boxFilter2DGPU<<<grid_size, block_size>>>(
        d_pitched_ptr, COLS, ROWS, DEPTH);

    int download_pixels[COLS][ROWS][DEPTH];
    d_parms.srcPtr.ptr = d_pitched_ptr.ptr;
    d_parms.srcPtr.pitch = d_pitched_ptr.pitch;
    d_parms.dstPtr.ptr = download_pixels;
    d_parms.dstPtr.pitch = COLS * sizeof(int);
    d_parms.kind = cudaMemcpyDeviceToHost;

    cudaMemcpy3D(&d_parms);

    for (int j = 0; j < ROWS; j++) {
        for (int i = 0; i < COLS; i++) {
            for (int k = 0; k < DEPTH; k++) {
                image.at<cv::Vec3b>(j, i)[k] = download_pixels[i][j][k];

                // std::cout << download_pixels[i][j][k]  << " ";
            }
            // std::cout << "\n";
        }
    }
}

