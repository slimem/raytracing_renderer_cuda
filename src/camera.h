#pragma once

#include "ray.h"

class camera {
public:
    __device__ camera(vec3 lookfrom, vec3 lookat, vec3 up,
        float vfov, float aspect,
        float aperture, float focus_dist,
        float time0 = 0.f, float time1 = 0.f) {

        _t0 = time0;
        _t1 = time1;

        _lens_radius = __fdiv_rz(aperture, 2);
        // vfov is top to bottom in degrees
        float theta = vfov * M_PI / 180.f;
        float half_height = __tanf(__fdiv_rz(theta, 2.f));
        float half_width = __fmul_rz(aspect, half_height);
        _origin = lookfrom;

        // get x y z unit camera vectors which are calculated using
        // lookfrom, look at and the up vector.
        _w = vec3::normalize(lookfrom - lookat);
        _u = vec3::normalize(vec3::cross(up, _w));
        _v = vec3::cross(_w, _u);

        _lowerLeft = _origin - half_width * focus_dist * _u - half_height * focus_dist * _v - focus_dist * _w;
        _horizontal = 2 * half_width * focus_dist * _u;
        _vertical = 2 * half_height * focus_dist * _v;
    }
    
    __device__ ray get_ray(float s, float t, curandState* rstate) {
        vec3 random = _lens_radius * utils::random_point_unit_disk(rstate);
        vec3 offset = _u * random.x() + _v * random.y(); // z is null
        float time = _t0 + curand_uniform(rstate) * (_t1 - _t0);
        return ray(_origin + offset, _lowerLeft + s * _horizontal + t * _vertical - _origin - offset, time);
    }

private:
    vec3 _origin;
    vec3 _lowerLeft;
    vec3 _horizontal;
    vec3 _vertical;
    vec3 _u, _v, _w;
    float _lens_radius;
    float _t0, _t1;
};
