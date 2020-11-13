#pragma once

#include "ray.h"

class hit_record {
public:
    __device__ hit_record() {};
    __device__ hit_record(float time, const vec3& position, const vec3& normal)
     : _t(time), _p(position), _n(normal) {};

    __device__ constexpr float t() const { return _t; }
    __device__ inline vec3 p() const { return _p; }
    __device__ inline vec3 n() const { return _n; }

    __device__ constexpr void set_t(float t) { _t = t; };
    __device__ inline void set_p(const vec3& p) { _p = p; }
    __device__ inline void set_n(const vec3& n) { _n = n; }

private:
    float _t;
    vec3 _p;
    vec3 _n;
};

// an object that can be hit with a ray
class hitable_object {
public:
    __device__ virtual bool hit(const ray& r, float tmin, float tmax, hit_record& hrec) const = 0;
};
