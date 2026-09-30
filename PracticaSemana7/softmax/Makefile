NVCC ?= nvcc
NVCCFLAGS ?= -O2 -std=c++17 -lineinfo
TARGET := softmax
SRC := main.cu
ROWS ?= 128
COLS ?= 1024

.PHONY: all run clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

run: $(TARGET)
	./$(TARGET) $(ROWS) $(COLS)

clean:
	rm -f $(TARGET)
