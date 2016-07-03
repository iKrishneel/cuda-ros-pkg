
#include <gpu_skeletonization/skeletonization_kernel.h>

#define CUDA_ERROR_CHECK(process) {                     \
      cudaAssert((process), __FILE__, __LINE__);        \
   }                                                    \

void cudaAssert(cudaError_t code, char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
       fprintf(stderr, "GPUassert: %s %s %dn",
               cudaGetErrorString(code), file, line);
       if (abort) {
          exit(code);
       }
    }
}

__host__ __device__ __align__(16)
int cuDivUp(int a, int b) {
    return ((a % b) != 0) ? (a / b + 1) : (a / b);
}


__global__
void skeletonizationKernel(unsigned char *d_image,
                           int iter, const int width, const int height) {
    int t_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int t_idy = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = t_idx + t_idy * blockDim.x * gridDim.x;
    
    unsigned int marker = 0;
    if ((t_idx > 0 && t_idx < width - 1) &&
        (t_idy > 0 && t_idy < height - 1)) {
       unsigned char val[9] = {};
       int icounter = 0;
       for (int y = -1; y <= 1; y++) {
          for (int x = -1; x <= 1; x++) {
             int index = (offset + x) + (width * y);
             val[icounter] = d_image[index];
             icounter++;
          }
       }
       
       int A = (val[3] == 0 && val[6] == 1) + (val[6] == 0 && val[7] == 1)
          + (val[7] == 0 && val[8] == 1) + (val[8] == 0 && val[5] == 1)
          + (val[5] == 0 && val[2] == 1) + (val[2] == 0 && val[1] == 1)
          + (val[1] == 0 && val[0] == 1) + (val[0] == 0 && val[3] == 1);
       int B  = val[3] + val[6] + val[7] + val[8]
          + val[5] + val[2] + val[1] + val[0];
       int m1 = iter == 0 ? (val[3] * val[7] * val[5])
          : (val[3] * val[7] * val[1]);
       int m2 = iter == 0 ? (val[5] * val[7] * val[1])
          : (val[3] * val[5] * val[1]);
       if (A == 1 && (B >= 2 && B <= 6) && m1 == 0 && m2 == 0) {
          marker = 1;
       } else {
          marker = 0;
       }
    }
    __syncthreads();
    
    if (offset < width * height) {
       d_image[offset] &= ~marker;
    }
}

__global__
void cuAbsDiff(int *value, unsigned char *d_image, unsigned char *d_prev,
               const int width, const int height) {
    int t_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int t_idy = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = t_idx + t_idy * blockDim.x * gridDim.x;

    if (offset == 0) {
       *value = 0;
    }
    __syncthreads();
    
    if (offset < width * height) {
       if (abs(d_image[offset] - d_prev[offset]) > 0) {
          atomicAdd(value, 1);
       }
    }
    __syncthreads();

    if (offset < width * height) {
       d_prev[offset] = d_image[offset];
    }
}

void skeletonizationGPU(cv::Mat &image) {
    if (image.type() != CV_8UC1) {
       cv::cvtColor(image, image, CV_BGR2GRAY);
    }
    const int im_size = image.rows * image.cols;
    unsigned char data[im_size];
    for (int j = 0; j < image.rows; j++) {
       for (int i = 0; i < image.cols; i++) {
          int index = i + (j * image.cols);
          data[index] = image.at<uchar>(j, i)/255;
       }
    }

    cudaEvent_t d_start;
    cudaEvent_t d_stop;
    cudaEventCreate(&d_start);
    cudaEventCreate(&d_stop);
    cudaEventRecord(d_start, 0);
    
    dim3 block_size(cuDivUp(image.cols, GRID_SIZE),
                    cuDivUp(image.rows, GRID_SIZE));
    dim3 grid_size(GRID_SIZE, GRID_SIZE);

    const int dev_mal = sizeof(unsigned char) * image.rows * image.cols;
    unsigned char *d_image;
    cudaMalloc(reinterpret_cast<void**>(&d_image), dev_mal);
    cudaMemcpy(d_image, data, dev_mal, cudaMemcpyHostToDevice);

    unsigned char *d_prev;
    cudaMalloc(reinterpret_cast<void**>(&d_prev), dev_mal);
    cudaMemset(d_prev, 0, dev_mal);
    
    int *d_count;
    cudaMalloc(reinterpret_cast<void**>(&d_count), sizeof(int));
    
    bool is_zero = true;
    unsigned char temp_data[im_size];
    int icounter = 0;
    
    int value;
    do {
       skeletonizationKernel<<<block_size, grid_size>>>(
          d_image, 0, image.cols, image.rows);
       cudaDeviceSynchronize();

       skeletonizationKernel<<<block_size, grid_size>>>(
          d_image, 1, image.cols, image.rows);
       cudaDeviceSynchronize();

       cuAbsDiff<<<block_size, grid_size>>>(
          d_count, d_image, d_prev, image.cols, image.rows);
       cudaMemcpy(&value, d_count, sizeof(int), cudaMemcpyDeviceToHost);

       if (value > 0) {
          is_zero = false;
       } else {
          is_zero = true;
          cudaMemcpy(temp_data, d_image, dev_mal, cudaMemcpyDeviceToHost);
       }

    } while (!is_zero);
    
    cudaEventRecord(d_stop, 0);
    cudaEventSynchronize(d_stop);
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, d_start, d_stop);

    std::cout << "\033[33m ELAPSED TIME:  \033[0m" << elapsed_time/1000.0f
              << "\n";
    
    icounter = 0;
    for (int i = 0; i < image.rows; i++) {
       for (int j = 0; j < image.cols; j++) {
          image.at<uchar>(i, j) = temp_data[icounter++] * 255;
       }
    }

    cudaFree(d_image);
    cudaFree(d_prev);
    cudaFree(d_count);
}

