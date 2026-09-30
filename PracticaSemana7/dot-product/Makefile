NVCC ?= nvcc
NVCCFLAGS ?= -O2 -std=c++17 -lineinfo
TARGET := dot_product
SRC := main.cu
N ?= 1048576

.PHONY: all run clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

run: $(TARGET)
	./$(TARGET) $(N)

clean:
	rm -f $(TARGET)
