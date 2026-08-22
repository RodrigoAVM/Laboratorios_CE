CC := gcc
CFLAGS := -Wall -Wextra -O3
LDFLAGS := -pthread

TARGETS := cpu-naive cpu-affinity

.PHONY: all clean

all: $(TARGETS)

cpu-naive: cpu-naive.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

cpu-affinity: cpu-affinity.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -f $(TARGETS)
