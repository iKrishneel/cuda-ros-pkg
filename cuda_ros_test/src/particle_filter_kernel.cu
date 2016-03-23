
#include <cuda_ros_test/particle_filter_kernel.h>

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

    dim3 dimBlock( blocksize, 1 );
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
    
    dim3 dim_thread(blocksize, blocksize);
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

__host__ __device__
struct cuImage{
    unsigned char pixel[3];
};

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

    dim3 dim_thread(blocksize, blocksize);
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
__host__ __device__
int iDivUp(int a, int b) {
    return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

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
    
    dim3 block_size(blocksize, blocksize);
    dim3 grid_size(iDivUp(COLS, blocksize), iDivUp(ROWS, blocksize));

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

