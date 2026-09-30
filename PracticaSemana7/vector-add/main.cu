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

__global__ void vector_add_kernel(const float *a, const float *b, float *c,
                                  int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        // TODO: Calcule c[i] = a[i] + b[i].
    }
}

static void fill_vectors(float *a, float *b, int n) {
    for (int i = 0; i < n; ++i) {
        a[i] = 0.5f * static_cast<float>(i);
        b[i] = 2.0f * static_cast<float>(i % 17);
    }
}

static bool verify_result(const float *a, const float *b, const float *c,
                          int n) {
    for (int i = 0; i < n; ++i) {
        float expected = a[i] + b[i];
        if (std::fabs(c[i] - expected) > 1e-5f) {
            std::fprintf(stderr,
                         "Error en indice %d: obtenido %.6f, esperado %.6f\n",
                         i, c[i], expected);
            return false;
        }
    }

    return true;
}

int main(int argc, char **argv) {
    int n = 1 << 20;
    if (argc > 1) {
        n = std::atoi(argv[1]);
    }

    size_t bytes = static_cast<size_t>(n) * sizeof(float);

    float *h_a = static_cast<float *>(std::malloc(bytes));
    float *h_b = static_cast<float *>(std::malloc(bytes));
    float *h_c = static_cast<float *>(std::malloc(bytes));

    if (!h_a || !h_b || !h_c) {
        std::fprintf(stderr, "No se pudo reservar memoria en CPU\n");
        return EXIT_FAILURE;
    }

    fill_vectors(h_a, h_b, n);

    float *d_a = nullptr;
    float *d_b = nullptr;
    float *d_c = nullptr;

    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_c, 0, bytes));

    int threads_per_block = 256;
    int blocks = (n + threads_per_block - 1) / threads_per_block;

    vector_add_kernel<<<blocks, threads_per_block>>>(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));

    bool ok = verify_result(h_a, h_b, h_c, n);
    std::printf("vector-add n=%d: %s\n", n, ok ? "OK" : "ERROR");

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    std::free(h_a);
    std::free(h_b);
    std::free(h_c);

    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
