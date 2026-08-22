#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define DEFAULT_THREADS 4
#define DEFAULT_MATRIX_SIZE 512
#define DEFAULT_TILE_SIZE 32
#define DEFAULT_REPETITIONS 10

static size_t min_size(size_t a, size_t b) {
  return a < b ? a : b;
}

static void init_matrix(double *matrix, size_t n, double scale) {
#pragma omp parallel for
  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < n; j++) {
      matrix[i * n + j] = scale * (double)((i + j) % 17 + 1);
    }
  }
}

static void zero_matrix(double *matrix, size_t n) {
#pragma omp parallel for
  for (size_t i = 0; i < n * n; i++) {
    matrix[i] = 0.0;
  }
}

static void matmul_tiled(const double *a, const double *b, double *c, size_t n,
                         size_t tile_size) {
  zero_matrix(c, n);

#pragma omp parallel for collapse(2) schedule(static)
  for (size_t ii = 0; ii < n; ii += tile_size) {
    for (size_t jj = 0; jj < n; jj += tile_size) {
      for (size_t kk = 0; kk < n; kk += tile_size) {
        size_t i_end = min_size(ii + tile_size, n);
        size_t j_end = min_size(jj + tile_size, n);
        size_t k_end = min_size(kk + tile_size, n);

        for (size_t i = ii; i < i_end; i++) {
          for (size_t k = kk; k < k_end; k++) {
            double a_value = a[i * n + k];

            for (size_t j = jj; j < j_end; j++) {
              c[i * n + j] += a_value * b[k * n + j];
            }
          }
        }
      }
    }
  }
}

static double checksum_matrix(const double *matrix, size_t n) {
  double sum = 0.0;

#pragma omp parallel for reduction(+ : sum)
  for (size_t i = 0; i < n * n; i++) {
    sum += matrix[i];
  }

  return sum;
}

static double *allocate_matrix(size_t n) {
  double *matrix = NULL;
  size_t bytes = n * n * sizeof(double);

  int ret = posix_memalign((void **)&matrix, 64, bytes);
  if (ret != 0) {
    return NULL;
  }

  return matrix;
}

int main(int argc, char **argv) {
  int num_threads = DEFAULT_THREADS;
  size_t matrix_size = DEFAULT_MATRIX_SIZE;
  size_t tile_size = DEFAULT_TILE_SIZE;
  int repetitions = DEFAULT_REPETITIONS;

  if (argc > 1) {
    num_threads = atoi(argv[1]);
  }

  if (argc > 2) {
    matrix_size = (size_t)atol(argv[2]);
  }

  if (argc > 3) {
    tile_size = (size_t)atol(argv[3]);
  }

  if (argc > 4) {
    repetitions = atoi(argv[4]);
  }

  if (num_threads <= 0 || matrix_size == 0 || tile_size == 0 ||
      repetitions <= 0) {
    fprintf(stderr, "Uso: %s [num_threads] [matrix_size] [tile_size] "
                    "[repeticiones]\n",
            argv[0]);
    return EXIT_FAILURE;
  }

  omp_set_num_threads(num_threads);

  double *a = allocate_matrix(matrix_size);
  double *b = allocate_matrix(matrix_size);
  double *c = allocate_matrix(matrix_size);

  if (a == NULL || b == NULL || c == NULL) {
    fprintf(stderr, "Error reservando memoria para matrices de %zux%zu\n",
            matrix_size, matrix_size);
    free(a);
    free(b);
    free(c);
    return EXIT_FAILURE;
  }

  init_matrix(a, matrix_size, 0.5);
  init_matrix(b, matrix_size, 0.25);

  double start = omp_get_wtime();

  for (int rep = 0; rep < repetitions; rep++) {
    matmul_tiled(a, b, c, matrix_size, tile_size);
  }

  double end = omp_get_wtime();
  double checksum = checksum_matrix(c, matrix_size);

  printf("Tamano matriz: %zux%zu\n", matrix_size, matrix_size);
  printf("Tamano tile: %zu\n", tile_size);
  printf("Threads solicitados: %d\n", num_threads);
  printf("Threads usados: %d\n", omp_get_max_threads());
  printf("Repeticiones: %d\n", repetitions);
  printf("Checksum: %.6f\n", checksum);
  printf("C[0][0]: %.6f\n", c[0]);
  printf("Tiempo: %.6f segundos\n", end - start);

  free(a);
  free(b);
  free(c);

  return EXIT_SUCCESS;
}
