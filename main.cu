#include <iostream>
#include <cassert>

#include "dp.hu"

#define CUDA_RUNTIME(stmt) checkCuda(stmt, __FILE__, __LINE__);

void checkCuda(cudaError_t result, const char *file, const int line) {
    if (result != cudaSuccess) {
      std::cerr << file << "@" << line << ": CUDA Runtime Error: " << cudaGetErrorString(result) << "\n";
      exit(-1);
    }
  }

int main(void) {

    float *a = nullptr;
    float *b = nullptr;
    float *s = nullptr;

    const size_t n = 100000;

    CUDA_RUNTIME(cudaMallocManaged(&a, n * sizeof(*a)));
    CUDA_RUNTIME(cudaMallocManaged(&b, n * sizeof(*b)));
    CUDA_RUNTIME(cudaMallocManaged(&s, n * sizeof(*s)));

    for (size_t i = 0; i < n; ++i) {
        a[i] = i;
        b[i] = 2*i;
    }

    launcher<<<1,1>>>(s,a,b,n);
    CUDA_RUNTIME(cudaGetLastError());
    CUDA_RUNTIME(cudaDeviceSynchronize());

    for (size_t i = 0; i < n; ++i) {
        assert(s[i] == a[i] + b[i]);
    }

    CUDA_RUNTIME(cudaFree(s));
    CUDA_RUNTIME(cudaFree(b));
    CUDA_RUNTIME(cudaFree(a));

    return 0;
}