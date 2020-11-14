#pragma once

#include "ray.h"

class camera {
public:
    __device__ camera()
     : _origin(vec3(0.f, 0.f, 0.f)),
       _lowerLeft(vec3(-2.f, -1.f, -1.f)),
       _horizontal(vec3(4.f, 0.f, 0.f)),
       _vertical(vec3(0.f, 2.f, 0.f))
    {}
    
    __device__ ray get_ray(float u, float v) {
        return ray(_origin, _lowerLeft + u * _horizontal + v * _vertical - _origin);
    }

private:
    vec3 _origin;
    vec3 _lowerLeft;
    vec3 _horizontal;
    vec3 _vertical;
};
