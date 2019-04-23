#include "dp.hu"

__global__ void vector_add(float *s, const float *a, const float *b, const size_t n) {
    for (size_t i = blockDim.x * blockIdx.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        s[i] = a[i] + b[i];
    }
}

__global__ void launcher(float *s, const float *a, const float *b, const size_t n) {
    vector_add<<<256,256>>>(s,a,b,n);
}