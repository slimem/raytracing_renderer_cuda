#include "common.h"
#include "vec3.h"
#include "ray.h"

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

__device__ bool hit_sphere(const vec3& center, float radius, const ray& r) {
    vec3 oc = r.origin() - center; // A - C
    // 1 - dot((p(​ t) - c)​ ,(p(​ t) - c​)) = R*R
    // 2 - dot((A​ + t*B ​- C)​ ,(A​ + t*B​ - C​)) = R*R (A is origin, B is direction)
    // 3 - t*t*dot(B,​ B)​ + 2*t*dot(B,A​-C​) + dot(A-C,A​-C​) - R*R = 0
    // we solve it as a 2nd degree polynomial with delta = b^2 - 4*a*c
    float a = vec3::dot(r.direction(), r.direction());
    float b = 2.f * vec3::dot(oc, r.direction());
    float c = vec3::dot(oc, oc) - radius * radius;
    float delta = b * b - 4 * a * c;
    return (delta > 0);
}

__device__ vec3 color(const ray& r) {
    if (hit_sphere(vec3(0.5f, -0.f, -0.8f), 0.3, r)) {
        return vec3(0.1f, 0.7f, 0.1f);
    }
    if (hit_sphere(vec3(0.f, 0.f, -1.f), 0.5, r)) {
        return vec3(0.1f, 0.1f, 0.7f);
    }
    if (hit_sphere(vec3(-0.5f, -0.f, -1.f), 0.3, r)) {
        return vec3(0.7f, 0.1f, 0.1f);
    }

    vec3 unit_direction = vec3::unit_vector(r.direction());
    float t = 0.5f * (unit_direction.y() + 1.f);
    //vec3 result = (1.f - t) * vec3(1.f, 1.f, 1.f) + t * vec3(0.5f, 0.7f, 1.f);
    vec3 result = (1.f - t) * vec3(1.f, 1.f, 1.f) + t * vec3(0.3f, 0.5f, 0.8f);
    // used for debug
    /*
        printf("OK: t = %f RDIR=%f,%f,%f UNITV=%f,%f,%f\n", t,
            r.direction().r(), r.direction().g(), r.direction().b(),
            unit_direction.r(), unit_direction.g(), unit_direction.b());
    }*/
    return result;
}

// we will divide the work on the GPU into blocks of 8x8 threads beacause
// 1 - can be multiplied to 32 so they can fit into warps easily
// 2 - is small so it helps similar pixels do similar work
#define THREAD_SIZE_X 8
#define THREAD_SIZE_Y 8

__global__ void render(vec3* frameBuffer, int width, int height,
    vec3 lowerLeftCorner, vec3 horizontal, vec3 vertical, vec3 origin) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    // if out of range
    if ((i >= width) || (j >= height)) {
        return;
    }
    int index = XY(i, j);
    float u = float(i) / float(width);
    float v = float(j) / float(height);
    ray r(origin, lowerLeftCorner + u * horizontal + v * vertical);
    frameBuffer[index] = color(r);
    // for debug purposes
    /*
    if (j % 2 && i % 2) {
        frameBuffer[index] = vec3(float(i) / width, float(j) / height, float(j) / (width + height));
    } else {
        frameBuffer[index] = vec3();
    }*/
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

    vec3 loweLeftCorner(-2.f, -1.f, -1.f);
    vec3 horizontal(4.f, 0.f, 0.f);
    vec3 vertical(0.f, 2.f, 0.f);
    vec3 origin(0.f, 0.f, 0.f);
    render<<<blocks, threads>>>(frameBuffer, WIDTH, HEIGHT,
        loweLeftCorner, horizontal, vertical, origin);

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