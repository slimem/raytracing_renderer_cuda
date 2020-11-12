#pragma once

#include "common.h"
#include "vec3.h"

// this class describes a ray.
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
    __device__ vec3 point_at_parameter(float t) const {
        return _a + t * _b;
    }

private:
    vec3 _a;
    vec3 _b;
};
