#pragma once

#include "common.h"
#include "vec3.h"

// this class describes a ray.
// All methods will be used by the device
class ray {
public:
    __device__ ray() {};
    __device__ ray(const vec3& origin, const vec3& direction)
        : _origin(origin), _direction(direction)
    {}
    __device__ const vec3& origin() const {
        return _origin;
    }
    __device__ const vec3& direction() const {
        return _direction;
    }
    __device__ vec3 point_at_parameter(float t) const {
        return _origin + t * _direction;
    }

private:
    vec3 _origin;
    vec3 _direction;
};
