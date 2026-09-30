#include <cmath>
#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                         cudaGetErrorString(err));                             \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                      \
    } while (0)

__global__ void dot_partial_kernel(const float *a, const float *b,
                                   float *partials, int n) {
    extern __shared__ float cache[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    float value = 0.0f;
    if (i < n) {
        // TODO: Calcule el producto local a[i] * b[i].
    }

    cache[tid] = value;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            // TODO: Acumule en cache[tid] el valor de cache[tid + stride].
        }
        __syncthreads();
    }

    if (tid == 0) {
        partials[blockIdx.x] = cache[0];
    }
}

static void fill_vectors(float *a, float *b, int n) {
    for (int i = 0; i < n; ++i) {
        a[i] = 1.0f + static_cast<float>(i % 13);
        b[i] = 0.25f * static_cast<float>((i % 7) - 3);
    }
}

static float cpu_dot_product(const float *a, const float *b, int n) {
    double acc = 0.0;
    for (int i = 0; i < n; ++i) {
        acc += static_cast<double>(a[i]) * static_cast<double>(b[i]);
    }

    return static_cast<float>(acc);
}

int main(int argc, char **argv) {
    int n = 1 << 20;
    if (argc > 1) {
        n = std::atoi(argv[1]);
    }

    int threads_per_block = 256;
    int blocks = (n + threads_per_block - 1) / threads_per_block;
    size_t bytes = static_cast<size_t>(n) * sizeof(float);
    size_t partial_bytes = static_cast<size_t>(blocks) * sizeof(float);

    float *h_a = static_cast<float *>(std::malloc(bytes));
    float *h_b = static_cast<float *>(std::malloc(bytes));
    float *h_partials = static_cast<float *>(std::malloc(partial_bytes));

    if (!h_a || !h_b || !h_partials) {
        std::fprintf(stderr, "No se pudo reservar memoria en CPU\n");
        return EXIT_FAILURE;
    }

    fill_vectors(h_a, h_b, n);

    float *d_a = nullptr;
    float *d_b = nullptr;
    float *d_partials = nullptr;

    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_partials, partial_bytes));

    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_partials, 0, partial_bytes));

    size_t shared_bytes = static_cast<size_t>(threads_per_block) * sizeof(float);
    dot_partial_kernel<<<blocks, threads_per_block, shared_bytes>>>(
        d_a, d_b, d_partials, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_partials, d_partials, partial_bytes,
                          cudaMemcpyDeviceToHost));

    double gpu_result = 0.0;
    for (int i = 0; i < blocks; ++i) {
        gpu_result += h_partials[i];
    }

    float cpu_result = cpu_dot_product(h_a, h_b, n);
    float error = std::fabs(static_cast<float>(gpu_result) - cpu_result);
    bool ok = error < 1e-2f;

    std::printf("dot-product n=%d: gpu=%.6f cpu=%.6f error=%.6f %s\n", n,
                static_cast<float>(gpu_result), cpu_result, error,
                ok ? "OK" : "ERROR");

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_partials));
    std::free(h_a);
    std::free(h_b);
    std::free(h_partials);

    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
