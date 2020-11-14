#include "common.h"
#include "vec3.h"
#include "ray.h"
#include "sphere.h"
#include "hitable_list.h"
#include "camera.h"
#include "curand_kernel.h"

// remember, the # converts the definition to a char*
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__)
//#define constructo

inline void check_cuda(cudaError_t errcode, char const* const func, const char* const file, int const line) {
    if (errcode) {
        fprintf(stderr, "check_cuda error (%d):\nFile \"%s\", line %d\n%s\n",
            static_cast<unsigned int>(errcode), file, line, cudaGetErrorString(errcode));
        cudaDeviceReset();
        exit(99);
    }
}

#define WIDTH 1200
#define HEIGHT 600

#define SAMPLES_PER_PIXEL 100

#define SEED 1000

// we will divide the work on the GPU into blocks of 8x8 threads beacause
// 1 - can be multiplied to 32 so they can fit into warps easily
// 2 - is small so it helps similar pixels do similar work
#define THREAD_SIZE_X 8
#define THREAD_SIZE_Y 8

// just to make things easier
__host__ __device__ constexpr int XY(int x, int y) {
    // change to intrinsic
    return y * WIDTH + x;
}

__device__ vec3 color(const ray& r, hitable_list** scene) {
    hit_record hrec;
    if ((*scene)->hit(r, 0.f, FLT_MAX, hrec)) {
        return 0.5f * vec3(hrec.n().x() + 1.0f, hrec.n().y() + 1.0f, hrec.n().z() + 1.0f);
    } else {
        vec3 unit_direction = vec3::normalize(r.direction());
        float t = 0.5f * (unit_direction.y() + 1.0f);
        return (1.0f - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
    }
}

__global__ void init_rand_state(curandState* randState, int width, int height) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    // if out of range
    if ((i >= width) || (j >= height)) {
        return;
    }

    int index = XY(i, j);
    
    // same seed for every thread
    curand_init(SEED, index, 0, &randState[index]);
}

__global__ void render(vec3* frameBuffer, int width, int height,
    hitable_list** scene,
    camera** cam,
    curandState* randState) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    // if out of range
    if ((i >= width) || (j >= height)) {
        return;
    }

    int index = XY(i, j);

    curandState rstate = randState[index];
    vec3 col;

    for (uint16_t sample = 0; sample < SAMPLES_PER_PIXEL; ++sample) {
        float u = float(i + curand_uniform(&rstate)) / float(width);
        float v = float(j + curand_uniform(&rstate)) / float(height);
        ray r = (*cam)->get_ray(u, v);
        col += color(r, scene);
    }

    frameBuffer[index] = col / float(SAMPLES_PER_PIXEL);
}

// TODO: check for array boundary
__global__ void populate_scene(hitable_object** objects, hitable_list** scene, camera** cam) {
    if (threadIdx.x == 0 && blockIdx.x == 0) { // only call once
        *(objects) = new sphere(vec3(0, 0, -1), 0.5);
        *(objects + 1) = new sphere(vec3(0, -100.5, -1), 100);
        *scene = new hitable_list(objects, 2);
        *cam = new camera();
    }
}

__global__ void free_scene(hitable_object** objects, hitable_list** scene, camera** cam) {
    // Objects already destoryed inside scene
    //delete* (objects);
    //delete* (objects + 1);
    delete* scene;
    delete* cam;
}

int main() {

    std::cout << "Rendering a " << WIDTH << "x" << HEIGHT << " image ";
    std::cout << "(" << SAMPLES_PER_PIXEL << " samples per pixel) ";
    std::cout << "in " << THREAD_SIZE_X << "x" << THREAD_SIZE_Y << " blocks.\n";

    // _d stands for device
    hitable_object** hitableObjects_d;
    hitable_list** scene_d;
    camera** camera_d;

    // random state
    curandState* rand_state_d;
    checkCudaErrors(cudaMalloc((void**)&rand_state_d, WIDTH * HEIGHT * sizeof(curandState)));

    // allocate unified memory that holds the size of our image
    vec3* frameBuffer_u; // u stands for unified
    size_t frameBufferSize = WIDTH * HEIGHT * sizeof(vec3); // RGB values for each pixel
    checkCudaErrors(cudaMallocManaged((void**)&frameBuffer_u, frameBufferSize));

    // allocate device memory
    checkCudaErrors(cudaMalloc((void**)&hitableObjects_d, 2 * sizeof(hitable_object*)));
    checkCudaErrors(cudaMalloc((void**)&scene_d, sizeof(hitable_list*)));
    checkCudaErrors(cudaMalloc((void**)&camera_d, sizeof(camera*)));

    // remember, construction is done in 1 block, 1 thread
    populate_scene<<<1, 1>>> (hitableObjects_d, scene_d, camera_d);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    auto start = std::chrono::system_clock::now();
    
    // remember: always round with + 1
    dim3 blocks(WIDTH / THREAD_SIZE_X + 1, HEIGHT / THREAD_SIZE_Y + 1);
    dim3 threads(THREAD_SIZE_X, THREAD_SIZE_Y);

    // init rand state for each pixel
    init_rand_state<<<blocks,threads>>>(rand_state_d, WIDTH, HEIGHT);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    render<<<blocks, threads>>>(frameBuffer_u, WIDTH, HEIGHT,
        scene_d,
        camera_d,
        rand_state_d);

    checkCudaErrors(cudaGetLastError());
    // block host until all device threads finish
    checkCudaErrors(cudaDeviceSynchronize());

    auto end = std::chrono::system_clock::now();

    auto timer_seconds = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout <<  "took " << timer_seconds << "us.\n";

    // Output frame buffer as a ppm image
    std::ofstream ppm_image("render.ppm");
    ppm_image << "P3\n" << WIDTH << " " << HEIGHT << "\n255\n";
    for (int j = HEIGHT - 1; j >= 0; j--) {
        for (int i = 0; i < WIDTH; i++) {
            size_t index = XY(i, j);
            float r = frameBuffer_u[index].r();
            float g = frameBuffer_u[index].g();
            float b = frameBuffer_u[index].b();
            int ir = int(255.99 * r);
            int ig = int(255.99 * g);
            int ib = int(255.99 * b);
            ppm_image << ir << " " << ig << " " << ib << "\n";
        }
    }
    ppm_image.close();

    // clean everything
    checkCudaErrors(cudaDeviceSynchronize());
    free_scene<<<1, 1>>>(hitableObjects_d, scene_d, camera_d);
    checkCudaErrors(cudaGetLastError());

    checkCudaErrors(cudaFree(hitableObjects_d));
    checkCudaErrors(cudaFree(scene_d));
    checkCudaErrors(cudaFree(camera_d));
    checkCudaErrors(cudaFree(rand_state_d));
    checkCudaErrors(cudaFree(frameBuffer_u));

    // Documentation: Destroy all allocations and reset all state on the
    // current device in the current process
    checkCudaErrors(cudaDeviceReset());

    return 0;
}