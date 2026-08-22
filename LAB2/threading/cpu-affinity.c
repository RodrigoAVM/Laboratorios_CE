#define _GNU_SOURCE

#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define NUM_THREADS 8
#define MEMORY_SIZE (256ULL * 1024ULL * 1024ULL)  // 256 MB por thread
#define NUM_ITERATIONS 10

typedef struct {
  int thread_id;
  int cpu_id;
  size_t memory_size;
} thread_arg_t;

static int pin_thread_to_cpu(int cpu_id) {
  cpu_set_t cpuset;

  CPU_ZERO(&cpuset);
  CPU_SET(cpu_id, &cpuset);

  return pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
}

/*
 * Kernel sencillo para consumir CPU.
 */
static void cpu_burn(double * restrict memory, size_t elements) {
  volatile double acc = 1.0;

  for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
    for (size_t i = 0; i < elements; i++) {
      double x = memory[i];

      /*
       * Suficiente calculo para mantener
       * ocupado al core.
       */
      x = x * 1.0000001 + acc;
      x = x * 0.9999999 + 0.000001;
      x = x * 1.0000003 + 0.000002;

      memory[i] = x;

      acc += x * 0.0000000001;
    }
  }
}

static void *worker(void *arg) {
  thread_arg_t *thread_arg = (thread_arg_t *)arg;

  int id = thread_arg->thread_id;
  int cpu = thread_arg->cpu_id;
  size_t memory_size = thread_arg->memory_size;

  int ret = pin_thread_to_cpu(cpu);
  if (ret != 0) {
    fprintf(stderr, "Thread %d: error fijando afinidad a CPU %d\n", id, cpu);
    return NULL;
  }

  double *memory = NULL;

  ret = posix_memalign((void **)&memory,
                       64,  // alineamiento cache-line
                       memory_size);

  if (ret != 0 || memory == NULL) {
    fprintf(stderr, "Thread %d: error reservando memoria\n", id);
    return NULL;
  }

  size_t elements = memory_size / sizeof(double);

  /*
   * First touch: al inicializar despues de fijar la afinidad, Linux intentara
   * asignar las paginas cerca del nodo NUMA donde corre este CPU.
   */
  for (size_t i = 0; i < elements; i++) {
    memory[i] = (double)(id + 1);
  }

  printf("Thread %d: %.2f MB inicializados desde CPU %d\n", id,
         memory_size / (1024.0 * 1024.0), sched_getcpu());

  cpu_burn(memory, elements);

  printf("Thread %d: finalizado en CPU %d\n", id, sched_getcpu());

  free(memory);

  return NULL;
}

int main(void) {
  pthread_t threads[NUM_THREADS];
  thread_arg_t args[NUM_THREADS];

  long num_cpus = sysconf(_SC_NPROCESSORS_ONLN);

  printf("CPUs logicos disponibles: %ld\n", num_cpus);

  if (num_cpus < NUM_THREADS) {
    fprintf(stderr, "Se necesitan al menos %d CPUs logicos\n", NUM_THREADS);
    return EXIT_FAILURE;
  }

  /*
   * CPU IDs que queremos utilizar.
   *
   * Puedes cambiarlos, por ejemplo:
   *
   * {0, 2, 4, 6}
   *
   * si quieres evitar hyperthreads hermanos.
   */
  int cpu_map[NUM_THREADS] = {0, 1, 2, 3, 4, 5, 6, 7};

  for (int i = 0; i < NUM_THREADS; i++) {
    args[i].thread_id = i;
    args[i].cpu_id = cpu_map[i];
    args[i].memory_size = MEMORY_SIZE;

    int ret = pthread_create(&threads[i], NULL, worker, &args[i]);

    if (ret != 0) {
      fprintf(stderr, "Error creando thread %d\n", i);
      return EXIT_FAILURE;
    }
  }

  for (int i = 0; i < NUM_THREADS; i++) {
    pthread_join(threads[i], NULL);
  }

  return EXIT_SUCCESS;
}
