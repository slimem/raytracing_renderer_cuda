#pragma once

#include "vec3.h"

class utils {
public:
    __device__ static vec3 random_point_unit_sphere(curandState* rstate);
};

__device__ vec3
utils::random_point_unit_sphere(curandState* rstate) {
    vec3 point;
    do {
        // grab a random point and center it in
        // the unit circle
        // the random value is generated using the random state
        // of the pixel calling the function
        point = 2.f * vec3(
            curand_uniform(rstate),
            curand_uniform(rstate),
            curand_uniform(rstate)
        ) - vec3(1.f, 1.f, 1.f);

    } while (point.sq_length() >= 1.f);
    return point;
}