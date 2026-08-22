#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define NUM_ELEMENTS 1000
#define DEFAULT_THREADS 4
#define DEFAULT_REPETITIONS 100000

static void init_input(double *x, size_t n) {
  for (size_t i = 0; i < n; i++) {
    x[i] = ((double)(i % 97) / 10.0) - 4.8;
  }
}

static void softmax(const double *x, double *y, size_t n) {
  double max_value = x[0];

#pragma omp parallel for reduction(max : max_value)
  for (size_t i = 0; i < n; i++) {
    if (x[i] > max_value) {
      max_value = x[i];
    }
  }

  double sum = 0.0;

#pragma omp parallel for reduction(+ : sum)
  for (size_t i = 0; i < n; i++) {
    y[i] = exp(x[i] - max_value);
    sum += y[i];
  }

#pragma omp parallel for
  for (size_t i = 0; i < n; i++) {
    y[i] /= sum;
  }
}

static double sum_array(const double *x, size_t n) {
  double sum = 0.0;

#pragma omp parallel for reduction(+ : sum)
  for (size_t i = 0; i < n; i++) {
    sum += x[i];
  }

  return sum;
}

int main(int argc, char **argv) {
  int num_threads = DEFAULT_THREADS;
  int repetitions = DEFAULT_REPETITIONS;

  if (argc > 1) {
    num_threads = atoi(argv[1]);
  }

  if (argc > 2) {
    repetitions = atoi(argv[2]);
  }

  if (num_threads <= 0 || repetitions <= 0) {
    fprintf(stderr, "Uso: %s [num_threads] [repeticiones]\n", argv[0]);
    return EXIT_FAILURE;
  }

  omp_set_num_threads(num_threads);

  double input[NUM_ELEMENTS];
  double output[NUM_ELEMENTS];

  init_input(input, NUM_ELEMENTS);

  double start = omp_get_wtime();

  for (int i = 0; i < repetitions; i++) {
    softmax(input, output, NUM_ELEMENTS);
  }

  double end = omp_get_wtime();
  double total = sum_array(output, NUM_ELEMENTS);

  printf("Elementos: %d\n", NUM_ELEMENTS);
  printf("Threads solicitados: %d\n", num_threads);
  printf("Threads usados: %d\n", omp_get_max_threads());
  printf("Repeticiones: %d\n", repetitions);
  printf("Suma softmax: %.17f\n", total);
  printf("Tiempo: %.6f segundos\n", end - start);
  printf("Primeros valores: %.12f %.12f %.12f %.12f %.12f\n", output[0],
         output[1], output[2], output[3], output[4]);

  return EXIT_SUCCESS;
}
