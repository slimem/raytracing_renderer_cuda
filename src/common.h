#pragma once

#include <iostream>
#include <fstream>
#include <chrono>
#define _USE_MATH_DEFINES
#include <math.h>
#include <cfloat>
#include "curand_kernel.h"

//#include "defines.h"

#define WIDTH 1200
#define HEIGHT 600

//#define WIDTH 16
//#define HEIGHT 8

#define RAY_BOUNCES 50
#define SEED 1000

// we will divide the work on the GPU into blocks of 8x8 threads beacause
// 1 - can be multiplied to 32 so they can fit into warps easily
// 2 - is small so it helps similar pixels do similar work
#define THREAD_SIZE_X 4
#define THREAD_SIZE_Y 4

#define STACK_SIZE 32