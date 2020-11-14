#pragma once

#include "vec3.h"

class utils {
public:
    __device__ static vec3 random_point_unit_sphere(curandState* rstate);

    __device__ static inline vec3 reflect(const vec3& v, const vec3& n);

    __device__ static inline bool refract(
        const vec3& v, const vec3& n,
        float ni_nt,
        vec3& refracted
    );

    //compute specular reflection coefficient R using Schlick's model
    __device__ static constexpr float shlick(float cosine, float refid);
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

__device__ inline vec3
utils::reflect(const vec3& v, const vec3& n) {
    // r = i - 2 ( i * n ) * n
    return v - 2.f * vec3::dot(v, n) * n;
}


// refract a vector using snells's law:
// i is the incident
// t is the refracted
// μ is n1/n2
// n is the normal vector
// t = μ * (i - (n.i)n) - n * sqrt(1 - μ*μ*(1 - (n.i)^2)
// the discriminant 1 - μ*μ*(1 - (n.i)^2 should be positive
__device__ inline bool
utils::refract(const vec3& v, const vec3& n,
    float mu,
    vec3& refracted) {
    //mu = n1/n2
    // n is already normalized
    vec3 i = vec3::normalize(v);
    float in = vec3::dot(i, n);
    float delta = 1.f - mu * mu * (1 - in * in);
    if (delta > 0) { // there is refraction
        refracted = mu * (i - n * in) - n * sqrt(delta);
        return true;
    } else {
        return false;
    }
}

__device__ constexpr float
utils::shlick(float cosine, float ref_id) {
#ifdef __CUDA_ARCH__

    float r0 = 
        __fdiv_rz(
        (1.f - ref_id),
        (1.f + ref_id)
        );
    r0 = __fmul_rz(r0, r0);
    return r0
        + __fmul_rz(
        (1.f - r0),
        __powf(1.f - cosine, 5.f)
        );
#else
    float r0 = (1.f - ref_id) / (1.f + ref_id);
    r0 = r0 * r0;
    return r0 + (1 - r0) * pow((1 - cosine), 5.f);
#endif
}