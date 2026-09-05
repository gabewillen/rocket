NVCC ?= nvcc
ARCH ?= sm_121
NVCCFLAGS ?= -O3 -arch=$(ARCH)

BIN = mem-bw mem-bw-sweep mem-bw-cachehint mem-gather gpu-capacity cpu-mem-contend

all: $(BIN)

mem-bw: mem-bw.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

mem-bw-sweep: mem-bw-sweep.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

mem-bw-cachehint: mem-bw-cachehint.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

mem-gather: mem-gather.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

gpu-capacity: gpu-capacity.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

cpu-mem-contend: cpu-mem-contend.c
	$(CC) -O2 -o $@ $< -lpthread

clean:
	rm -f $(BIN)

.PHONY: all clean
