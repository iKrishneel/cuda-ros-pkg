
// #include <gpu_particle_filter/particle_filter_kernel.h>

typedef struct __align__(16) {
    int width;
    int height;
    int stride;
    float *elements;
} cuMat;

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
cuMat MatrixProduct(cuMat A, cuMat B, cuMat c) {
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