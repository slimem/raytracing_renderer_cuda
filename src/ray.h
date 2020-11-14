#pragma once

#include "common.h"
#include "vec3.h"

// this class describes a ray in 3D space with
// an origin and a direction.
// All methods will be used by the device
class ray {
public:
    __device__ ray() {};
    __device__ ray(const vec3& a, const vec3& b)
        : _a(a), _b(b)
    {}
    __device__ const vec3& origin() const {
        return _a;
    }
    __device__ const vec3& direction() const {
        return _b;
    }

    // for each parameter t, we get the coordinates
    // of the point in 3D space. Think of it as an
    // affine equation, and each x gives you a new
    // point in 3D space
    __device__ vec3 point_at_parameter(float t) const {
        return _a + t * _b;
    }

private:
    vec3 _a; // ray origin x y z coordinates
    vec3 _b; // ray direction
};
