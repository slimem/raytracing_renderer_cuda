CUDA_PATH?=/usr/local/cuda
HOST_COMPILER=g++
#cpp17 is not supported yet for gcc host compiler
CPP_VERSION=-std=c++14 
NVCC=$(CUDA_PATH)/bin/nvcc $(CPP_VERSION) -ccbin $(HOST_COMPILER)

NVPROF_FLAGS=--metrics achieved_occupancy,inst_executed,inst_fp_32,inst_fp_64,inst_integer

# debug vs release
NVCCFLAGS_DEBUG= -g -G
NVCCFLAGS=

all: clean main.o compile run
debug: clean dbg

compile: main.o
	$(NVCC) $(NVCCFLAGS) -o main main.o

main.o:
	$(NVCC) $(NVCCFLAGS) -o main.o -c src/main.cu

dbg:
	$(NVCC) $(NVCCFLAGS_DEBUG) -o main.o -c src/main.cu
	$(NVCC) $(NVCCFLAGS_DEBUG) -o main main.o

run: main
	rm -f render.jpg
	./main

profile: main
	nvprof ./main

profile_metrics: main
	nvprof $(NVPROF_FLAGS) ./main

clean:
	rm -f main main.o render.jpg

