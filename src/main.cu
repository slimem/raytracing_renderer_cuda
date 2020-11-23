#include "common.h"
#include "vec3.h"
#include "ray.h"
#include "sphere.h"
#include "hitable_list.h"
#include "camera.h"
#include "bvh.h"
#include "texture.h"

#define STB_IMAGE_IMPLEMENTATION
#include "libs/stb/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "libs/stb/stb_image_write.h"

#define SAMPLES_PER_PIXEL 100

// remember, the # converts the definition to a char*
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__)

inline void check_cuda(cudaError_t errcode, char const* const func, const char* const file, int const line) {
    if (errcode) {
        fprintf(stderr, "check_cuda error (%d):\nFile \"%s\", line %d\n%s\n",
            static_cast<unsigned int>(errcode), file, line, cudaGetErrorString(errcode));
        cudaDeviceReset();
        exit(99);
    }
}

texture<float, 2, cudaReadModeElementType> tex;
constexpr char imagePath[] = "textures/earth.jpg";

__global__ void transformKernel(float* outputData,
    int width,
    int height,
    float theta)
{
    // calculate normalized texture coordinates
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    float u = (float)x - (float)width / 2;
    float v = (float)y - (float)height / 2;
    float tu = u * cosf(theta) - v * sinf(theta);
    float tv = v * cosf(theta) + u * sinf(theta);

    tu /= (float)width;
    tv /= (float)height;

    // read from texture and write to global memory
    outputData[y * width + x] = tex2D(tex, tu + 0.5f, tv + 0.5f);
}


__device__ vec3 color(const ray& r, hitable_list** scene, curandState* rstate) {

    // this section is a simple implementation for a diffuse material with a 50%
    // attenuation at each bounce
    ray curr_r = r;
    vec3 curr_attenuation(1.f, .8f, .7f);
    //vec3 curr_attenuation(0.067, 0.471, 0.576);
    for (int i = 0; i < RAY_BOUNCES; ++i) {
        hit_record hrec;
        // 0.001 -> ignore hits near zero
        if ((*scene)->hit(curr_r, 0.00001f, FLT_MAX, hrec)) {
            ray scattered;
            vec3 attenuation;

            vec3 emit = hrec.m()->emit() + vec3(0.1,0.1,0.1); // bloomy effect
            if (hrec.m()->scatter(curr_r, scattered, hrec, attenuation, rstate)) {
                curr_attenuation = emit + attenuation*curr_attenuation;
                curr_r = scattered;
            } else {
                return emit;
            }

            /*vec3 target = hrec.p() + hrec.n() + random_point_unit_sphere(rstate);
            curr_attenuation *= 0.5f;
            curr_r = ray(hrec.p(), target - hrec.p());*/
        } else {
            /*vec3 unit_direction = vec3::normalize(curr_r.direction());
            float t = 0.5f * (unit_direction.y() + 1.0f);
            vec3 v = (1.0f - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
            return curr_attenuation * v;
            */
            // return world color
            return curr_attenuation;
        }
    }
    return vec3(); // exceeded recursion
    /*if ((col.r() < 0) || (col.g() < 0) || (col.b() < 0)) {
    printf("ERROR: COL=%f,%f,%f\n", col.r(), col.g(), col.b());
    }*/
}

__global__ void init_rand_state(curandState* randState, int width, int height) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    // if out of range
    if ((i >= width) || (j >= height)) {
        return;
    }

    int index = utils::XY(i, j);
    
    // same seed for every thread, very slow
    //curand_init(SEED, index, 0, &randState[index]);

    // different seed for each thread, fast
    curand_init(SEED + index, 0, 0, &randState[index]);

    // produces weird artifacts
    //curand_init(SEED, 0, 0, &randState[index]);
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

    int index = utils::XY(i, j);

    curandState rstate = randState[index];
    vec3 col;

    for (uint16_t sample = 0; sample < SAMPLES_PER_PIXEL; ++sample) {
        // remember: random value is [0, 1[ 
        float u = float(i + curand_uniform(&rstate)) / float(width);
        float v = float(j + curand_uniform(&rstate)) / float(height);
        ray r = (*cam)->get_ray(u, v, &rstate);
        col += color(r, scene, &rstate);
        
    }

    
    col /= float(SAMPLES_PER_PIXEL);
        //col.saturate();
    // do gamma correction with gamma 2 => raise the color to the power of 1/2 (sqrt)
    frameBuffer[index] = col.saturate().gamma_correct();

    // only for debug
    //frameBuffer[index] = col.gamma_correct();

}

// TODO: check for array boundary
__global__ void populate_scene(hitable_object** objects, hitable_list** scene,
    camera** cam, curandState* state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) { // only call once
        // sphere 1
        objects[0] = new sphere(
            vec3(0, 0, -1),
            0.5,
            new lambertian(new constant_texture(vec3(0.6, 0.1, 0.1)))
            //new dielectric(1.3, vec3(1, 1, 1))
            //new dielectric(1.5, vec3(1,1,1))
        );

        objects[0]->set_id(0);
        
        

        // sphere 2


        /*text* checker = new checker_texture(
            new constant_texture(vec3(0.1, 0.2, 0.5)),
            new constant_texture(vec3(0.5, 0.2, 0.1)));*/
        //text* noise1 = new noise_texture(noise_type::TURBULANCE, .1f);
        text* noise1 = new noise_texture(noise_type::MARBLE, 1.f);
        /*text* noise = new wood_texture(vec3(0.792, 0.643, 0.447),
            //vec3(0.267, 0.188, 0.133),
            vec3(0.412, 0.349, 0.306),
            10.f);*/
        /*text* checker = new checker_texture(
            noise,
            noise);*/

        //text* noise1 = new noise_texture(noise_type::MARBLE, .2f);
        objects[1] = new sphere(
            vec3(0, -1000.5, 1),
            1000,
            new lambertian(noise1)
            //new lambertian(new constant_texture(vec3(0.1, 0.2, 0.5)))
        );
        objects[1]->set_id(1);
        /*objects[1] = new sphere(
            vec3(0, -20, 1),
            10,
            new lambertian(vec3(0.1, 0.2, 0.5))
        );
        objects[1]->set_id(1);*/


        //text* im_text = new image_texture(nullptr, 200, 300);
        //sphere 3
        objects[2] = new sphere(
            vec3(1, 0, -1),
            0.5,
            //new dielectric(1.5)
            //new lambertian(noise1)
            new lambertian(new constant_texture(vec3(0.1, 0.2, 0.5)))
            //new metal(vec3(1.f), 0.f)
            //new metal(vec3(1.f), 0.f)
            //new metal(vec3(0.075, 0.461, 0.559), 0.1f)
        );
        objects[2]->set_id(2);

        //sphere 4

        //perlin_noise::init(state);
        //perlin_noise noise;
        //text* per_text = new noise_texture(state);

        objects[3] = new sphere(
            vec3(-1, 0, -2),
            0.5,

            //new lambertian(per_text)
            new metal(vec3(1.f), 0.f)
//            new lambertian(new constant_texture(vec3(0.6, 0.1, 0.1)))
            //new dielectric(1.5, vec3(1, 1, 1))
            //new metal(vec3(0.8, 0.8, 0.8), 0.5)
        );
        objects[3]->set_id(3);

        //sphere 5
        objects[4] = new sphere(
            vec3(0, 0, -2),
            0.5,
            new metal(vec3(0.8, 0.8, 0.8), 0.5)
            //new metal(vec3(0.8, 0.8, 0.8), 0.5)
        );
        objects[4]->set_id(4);
        
        objects[5] = new sphere(
            vec3(1, 0, -2),
            0.5,
            //new emitter(vec3(1,0.5,0.5))

            new dielectric(1.5, vec3(1, 1, 1))
            //new lambertian(new constant_texture(vec3(0.1, 0.2, 0.5)))
            //new lambertian(vec3(0.2, 0.9, 0.3)*0.6)
            
        );
        objects[5]->set_id(5);

        
        objects[6]= new sphere(
            vec3(-1, 0, -1),
            0.5,
            new emitter(vec3(0.5,1,0.5))
            //new dielectric(1.1, vec3(0.8,1.0,0.8))
        );
        objects[6]->set_id(6);

        objects[7] = new moving_sphere(
            vec3(-1, 1, -1),
            vec3(-2, 1, -1),
            0.f,
            1.f,
            0.2,

            new lambertian(new constant_texture(vec3(0.6, 0.1, 0.1)))
            //new dielectric(1.5, vec3(1, 1, 1))
            //new metal(vec3(0.8, 0.8, 0.8), 0.5)
        );
        /*objects[7] = new sphere(
            vec3(-1, 1, -1),
            0.5,
            new lambertian(vec3(0.6, 0.1, 0.1)));*/
        objects[7]->set_id(7);

        objects[8] = new bvh_node(objects, 8, 0, 1, state, 0);
        objects[8]->set_id(8);

        // check bvh hierarchy
        //bvh_node::display_tree(static_cast<bvh_node*>(objects[8]), 2);

        *scene = new hitable_list(objects, static_cast<bvh_node*>(objects[8]), 8);
        scene[0]->set_id(9);

        //for (int i = 0; i < 9; ++i) {
        //    printf("(%d) %s\n", objects[i]->get_id(), hitable_object::obj_type_str(objects[i]->get_object_type()));
        //}

        //vec3 lookfrom = vec3(-2, 1, 2) * 2;
        vec3 lookfrom = vec3(-1, 2,5);
        //THISvec3 lookfrom = vec3(5, 2, 3);
        //vec3 lookat = vec3(0, 0, -1);
        //vec3 lookat = vec3(-1, 0, -1); // redball
        //vec3 lookat = vec3(1, 0, -1); // marble ball
        vec3 lookat = vec3(0, 0, -1);
        float dist_to_focus = (lookfrom - lookat).length();
        float aperture = .25f;
        *cam = new camera(
            lookfrom, // lookfrom
            lookat, // lookat
            vec3(0,1,0),   // up
            20.f,           // fov
            float(WIDTH) / float(HEIGHT),
            aperture,
            dist_to_focus,
            0,
            0.2
        );

        //hit_record hrec;
        //ray r = (*cam)->get_ray(0.54, 0.5, state);
        //static_cast<bvh_node*>(objects[8])->dfs(r, 0.001f, FLT_MAX, hrec);
        //assert(0);
    }
}

__global__ void free_scene(hitable_object** objects, hitable_list** scene, camera** cam) {
    // Objects already destoryed inside scene
    //delete* (objects);
    //delete* (objects + 1);
    delete* scene;
    delete* cam;
}

int main(int argc, char** argv) {

    // loading image to host
    int w, h, ch;
    uint8_t* imgData = stbi_load(imagePath, &w, &h, &ch, 0);
    std::cout << "Loaded image with " << w << "x" << h << " and " << ch << " channels\n";
    stbi_write_png("export.png", w, h, ch, imgData, w * ch);
    stbi_image_free(imgData);

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
    checkCudaErrors(cudaMalloc((void**)&hitableObjects_d, 9 * sizeof(hitable_object*)));
    checkCudaErrors(cudaMalloc((void**)&scene_d, sizeof(hitable_list*)));
    checkCudaErrors(cudaMalloc((void**)&camera_d, sizeof(camera*)));

    // remember, construction is done in 1 block, 1 thread
    populate_scene<<<1, 1>>> (hitableObjects_d, scene_d, camera_d, rand_state_d);
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
#if 0
    std::ofstream ppm_image("render.ppm");
    ppm_image << "P3\n" << WIDTH << " " << HEIGHT << "\n255\n";
    for (int j = HEIGHT - 1; j >= 0; j--) {
        for (int i = 0; i < WIDTH; i++) {
            size_t index = utils::XY(i, j);
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
#endif

    uint8_t* imgBuff = (uint8_t*)std::malloc(WIDTH * HEIGHT * 3 * sizeof(uint8_t));
    for (int j = HEIGHT - 1; j >= 0; --j) {
        for (int i = 0; i < WIDTH; ++i) {
            size_t index = utils::XY(i, j);
            // stbi generates a Y flipped image
            size_t rev_index = utils::XY(i, HEIGHT - j - 1);
            float r = frameBuffer_u[index].r();
            float g = frameBuffer_u[index].g();
            float b = frameBuffer_u[index].b();
            imgBuff[rev_index * 3 + 0] = int(255.999f * r) & 255;
            imgBuff[rev_index * 3 + 1] = int(255.999f * g) & 255;
            imgBuff[rev_index * 3 + 2] = int(255.999f * b) & 255;
        }
    }
    //stbi_write_png("render.png", WIDTH, HEIGHT, 3, imgBuff, WIDTH * 3);
    stbi_write_jpg("render.jpg", WIDTH, HEIGHT, 3, imgBuff, 100);
    std::free(imgBuff);

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