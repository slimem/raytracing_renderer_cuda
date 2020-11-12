#include "common.h"
#include "vec3.h"

// checkCudaErrors from helper_cuda.h in CUDA examples
// remember, the # converts the definition to a char*
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__)

void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) {
    if (result) {
        std::cerr << "[CUDA ERROR " << static_cast<unsigned int>(result) << "] : " <<
            file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

#define WIDTH 1200
#define HEIGHT 600

// just to make things easier
__host__ __device__ constexpr int XY(int x, int y) {
    return y * WIDTH + x;
}

// we will divide the work on the GPU into blocks of 8x8 threads beacause
// 1 - can be multiplied to 32 so they can fit into warps easily
// 2 - is small so it helps similar pixels do similar work
#define THREAD_SIZE_X 8
#define THREAD_SIZE_Y 8

__global__ void render(vec3* frameBuffer, int width, int height) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    // if out of range
    if ((i >= width) || (j >= height)) {
        return;
    }

    int index = XY(i, j);
    if (j % 2 && i % 2) {
        frameBuffer[index] = vec3(float(i) / width, float(j) / height, float(j) / (width + height));
    } else {
        frameBuffer[index] = vec3();
    }
}

int main() {

    std::cerr << "Rendering a " << WIDTH << "x" << HEIGHT << " image ";
    std::cerr << "in " << THREAD_SIZE_X << "x" << THREAD_SIZE_Y << " blocks.\n";

    // RGB values for each pixel
    size_t frameBufferSize = WIDTH * HEIGHT * sizeof(vec3);

    vec3* frameBuffer;
    // allocate unified memory that holds the size of our image
    // remember, cudaMallocManaged waits for void**
    checkCudaErrors(cudaMallocManaged((void**)&frameBuffer, frameBufferSize));

    auto start = std::chrono::system_clock::now();
    
    // remember: always round with + 1
    dim3 blocks(WIDTH / THREAD_SIZE_X + 1, HEIGHT / THREAD_SIZE_Y + 1);
    dim3 threads(THREAD_SIZE_X, THREAD_SIZE_Y);

    render<<<blocks, threads>>>(frameBuffer, WIDTH, HEIGHT);

    checkCudaErrors(cudaGetLastError());
    // block host until all device threads finish
    checkCudaErrors(cudaDeviceSynchronize());

    auto end = std::chrono::system_clock::now();

    auto timer_seconds = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cerr <<  "took " << timer_seconds << "us.\n";

    // Output frame buffer as a ppm image 
    std::cout << "P3\n" << WIDTH << " " << HEIGHT << "\n255\n";
    for (int j = HEIGHT - 1; j >= 0; j--) {
        for (int i = 0; i < WIDTH; i++) {
            size_t index = XY(i, j);
            float r = frameBuffer[index].r();
            float g = frameBuffer[index].g();
            float b = frameBuffer[index].b();
            int ir = int(255.99 * r);
            int ig = int(255.99 * g);
            int ib = int(255.99 * b);
            std::cout << ir << " " << ig << " " << ib << "\n";
        }
    }

    checkCudaErrors(cudaFree(frameBuffer));

    return 0;
}