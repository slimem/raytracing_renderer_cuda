#pragma once

#include "ray.h"

class camera {
public:
    __device__ camera(vec3 lookfrom, vec3 lookat, vec3 up,
        float vfov, float aspect) {
        // vfov is top to bottom in degrees
        float theta = vfov * M_PI / 180.f;
        float half_height = __tanf(__fdiv_rz(theta, 2.f));
        float half_width = __fmul_rz(aspect, half_height);
        _origin = lookfrom;

        // get x y z unit camera vectors which are calculated using
        // lookfrom, look at and the up vector.
        vec3 w = vec3::normalize(lookfrom - lookat);
        vec3 u = vec3::normalize(vec3::cross(up, w));
        vec3 v = vec3::cross(w, u);

        _lowerLeft = _origin - half_width * u - half_height * v - w;
        _horizontal = 2 * half_width * u;
        _vertical = 2 * half_height * v;
    }
    
    __device__ ray get_ray(float u, float v) {
        return ray(_origin, _lowerLeft + u * _horizontal + v * _vertical - _origin);
    }

private:
    vec3 _origin;
    vec3 _lowerLeft;
    vec3 _horizontal;
    vec3 _vertical;
};
