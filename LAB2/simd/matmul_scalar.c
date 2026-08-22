#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define VECTOR_SIZE 2048
#define MATRIX_SIZE 2048
#define DEFAULT_REPETITIONS 1

float dot_product_scalar(const float a[VECTOR_SIZE], const float b[VECTOR_SIZE])
{
    float result = 0.0f;

    for (int i = 0; i < VECTOR_SIZE; ++i) {
        result += a[i] * b[i];
    }

    return result;
}

void transpose_matrix_scalar(
    const float matrix[MATRIX_SIZE][MATRIX_SIZE],
    float transposed[MATRIX_SIZE][MATRIX_SIZE])
{
    for (int row = 0; row < MATRIX_SIZE; ++row) {
        for (int col = 0; col < MATRIX_SIZE; ++col) {
            transposed[col][row] = matrix[row][col];
        }
    }
}

void matrix_multiply_scalar(
    const float a[MATRIX_SIZE][MATRIX_SIZE],
    const float b[MATRIX_SIZE][MATRIX_SIZE],
    float result[MATRIX_SIZE][MATRIX_SIZE])
{
    static float b_transposed[MATRIX_SIZE][MATRIX_SIZE];

    transpose_matrix_scalar(b, b_transposed);

    for (int row = 0; row < MATRIX_SIZE; ++row) {
        for (int col = 0; col < MATRIX_SIZE; ++col) {
            result[row][col] = dot_product_scalar(a[row], b_transposed[col]);
        }
    }
}

static double get_time_seconds(void)
{
    struct timespec time;

    clock_gettime(CLOCK_MONOTONIC, &time);

    return (double)time.tv_sec + (double)time.tv_nsec / 1000000000.0;
}

static void init_matrix_scalar(float matrix[MATRIX_SIZE][MATRIX_SIZE], float scale)
{
    for (int row = 0; row < MATRIX_SIZE; ++row) {
        for (int col = 0; col < MATRIX_SIZE; ++col) {
            matrix[row][col] = scale * (float)((row + col) % 17 + 1);
        }
    }
}

static double checksum_matrix_scalar(const float matrix[MATRIX_SIZE][MATRIX_SIZE])
{
    double sum = 0.0;

    for (int row = 0; row < MATRIX_SIZE; ++row) {
        for (int col = 0; col < MATRIX_SIZE; ++col) {
            sum += matrix[row][col];
        }
    }

    return sum;
}

static float matrix_a[MATRIX_SIZE][MATRIX_SIZE];
static float matrix_b[MATRIX_SIZE][MATRIX_SIZE];
static float matrix_result[MATRIX_SIZE][MATRIX_SIZE];

int main(int argc, char **argv)
{
    int repetitions = DEFAULT_REPETITIONS;

    if (argc > 1) {
        repetitions = atoi(argv[1]);
    }

    if (repetitions <= 0) {
        fprintf(stderr, "Uso: %s [repeticiones]\n", argv[0]);
        return EXIT_FAILURE;
    }

    init_matrix_scalar(matrix_a, 0.5f);
    init_matrix_scalar(matrix_b, 0.25f);

    double start = get_time_seconds();

    for (int rep = 0; rep < repetitions; ++rep) {
        matrix_multiply_scalar(matrix_a, matrix_b, matrix_result);
    }

    double end = get_time_seconds();
    double elapsed = end - start;
    double operations = (double)repetitions * 2.0 * MATRIX_SIZE * MATRIX_SIZE * MATRIX_SIZE;
    double gflops = operations / elapsed / 1000000000.0;
    double checksum = checksum_matrix_scalar(matrix_result);

    printf("Tamano matriz: %dx%d\n", MATRIX_SIZE, MATRIX_SIZE);
    printf("Repeticiones: %d\n", repetitions);
    printf("Checksum: %.6f\n", checksum);
    printf("C[0][0]: %.6f\n", matrix_result[0][0]);
    printf("C[1023][1023]: %.6f\n", matrix_result[1023][1023]);
    printf("Operaciones: %.0f\n", operations);
    printf("Tiempo: %.6f segundos\n", elapsed);
    printf("Rendimiento: %.6f GFLOP/s\n", gflops);

    return EXIT_SUCCESS;
}
