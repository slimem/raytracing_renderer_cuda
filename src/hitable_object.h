#pragma once

#include "ray.h"

enum class hit_object_type {
    SPHERE,
    UNKNOWN
};

// Every hitable_object will hold a hit_record that contains
// the hit point _p, the normal vector _n and a t parameter
// value
class hit_record {
public:
    __device__ hit_record() {};
    __device__ hit_record(
        float parameter,
        const vec3& position,
        const vec3& normal,
        const hit_object_type& otype = hit_object_type::UNKNOWN)
     : _t(parameter), _p(position), _n(normal), _h(otype) {};

    __device__ constexpr float t() const { return _t; }
    __device__ inline vec3 p() const { return _p; }
    __device__ inline vec3 n() const { return _n; }
    __device__ constexpr hit_object_type h() const { return _h; }

    __device__ constexpr void set_t(float t) { _t = t; };
    __device__ inline void set_p(const vec3& p) { _p = p; }
    __device__ inline void set_n(const vec3& n) { _n = n; }
    __device__ constexpr void set_h(const hit_object_type& h) {
        _h = h;
    }

private:
    float _t;
    vec3 _p;
    vec3 _n;
    hit_object_type  _h = hit_object_type::UNKNOWN;
};

// Abstract class for any object that can be hit with a ray r
class hitable_object {
public:
    __device__ virtual bool hit(const ray& r, float tmin, float tmax, hit_record& hrec) const = 0;
};
