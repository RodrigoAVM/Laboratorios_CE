CC := gcc
CFLAGS := -Wall -Wextra -O3
AVX2FLAGS := -mavx2

TARGETS := matmul_avx2 matmul_scalar
REPETITIONS ?= 1

.PHONY: all run run-avx2 run-scalar clean

all: $(TARGETS)

matmul_avx2: matmul_avx2.c
	$(CC) $(CFLAGS) $(AVX2FLAGS) $< -o $@

matmul_scalar: matmul_scalar.c
	$(CC) $(CFLAGS) $< -o $@

run: run-avx2

run-avx2: matmul_avx2
	./matmul_avx2 $(REPETITIONS)

run-scalar: matmul_scalar
	./matmul_scalar $(REPETITIONS)

clean:
	rm -f $(TARGETS)
