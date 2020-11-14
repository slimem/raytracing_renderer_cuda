CUDA_PATH?=/usr/local/cuda
HOST_COMPILER=g++
#cpp17 is not supported yet for gcc host compiler
CPP_VERSION=-std=c++14 
NVCC=$(CUDA_PATH)/bin/nvcc $(CPP_VERSION) -ccbin $(HOST_COMPILER)

# debug vs release
NVCCFLAGS=

all: clean main.o compile run convert

compile: main.o
	$(NVCC) -o main main.o

main.o:
	$(NVCC) -o main.o -c src/main.cu

run: main
	rm -f render.ppm
	./main

convert: render.ppm
	ppmtojpeg render.ppm > render.jpg

profile: main
	nvprof ./main > render.ppm

clean:
	rm -f main main.o render.ppm render.jpg

