#pragma once

#include "ray.h"
#include "aabb.h"

class material;

enum class hit_object_type {
    SPHERE,
    UNKNOWN
};

// since we cannot have dynamic cast
enum class object_type {
    SPHERE,
    MOVING_SPHERE,
    HITABLE_LIST,
    BOUNDING_VOLUME_HIERARCHY,
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
        const material* mat, //use shared pointer
        const hit_object_type& otype = hit_object_type::UNKNOWN)
      : _t(parameter),
        _p(position),
        _n(normal),
        _h(otype),
        _m(mat)
    {};

    __device__ constexpr float t() const { return _t; }
    __device__ constexpr float u() const { return _u; }
    __device__ constexpr float v() const { return _v; }
    __device__ inline vec3 p() const { return _p; }
    __device__ inline vec3 n() const { return _n; }
    __device__ constexpr hit_object_type h() const { return _h; }
    __device__ constexpr const material* m() const { return _m; }

    __device__ constexpr void set_t(float t) { _t = t; };
    __device__ constexpr void set_u(float u) { _u = u; };
    __device__ constexpr void set_v(float v) { _v = v; };
    __device__ inline void set_p(const vec3& p) { _p = p; }
    __device__ inline void set_n(const vec3& n) { _n = n; }
    __device__ constexpr void set_m(const material* mat) { _m = mat;  }
    __device__ constexpr void set_h(const hit_object_type& h) { _h = h; }

private:
    float _t = 0;
    float _u = 0, _v = 0;
    vec3 _p;
    vec3 _n;
    const material* _m = nullptr; // the hit record is not allowed to change the material
    hit_object_type  _h = hit_object_type::UNKNOWN;
};

// Abstract class for any object that can be hit with a ray r
class hitable_object {
public:
    __device__ virtual bool hit(const ray& r, float tmin, float tmax, hit_record& hrec) { return false; };
    __device__ virtual bool bounding_box(float t0, float t1, AABB& box) const { return false; };
    __device__ virtual object_type get_object_type() const { return object_type::UNKNOWN; }
    __device__ inline bool is_leaf() const {
        return get_object_type() != object_type::BOUNDING_VOLUME_HIERARCHY;
    }
    __device__ virtual ~hitable_object() noexcept {}
    __device__ static const char* obj_type_str(object_type obj);

    __device__ constexpr uint32_t get_id() const { return _id; };
    __device__ constexpr void set_id(const uint32_t id) { _id = id; }

private:
    uint32_t _id = 0;
};

__device__ const char*
hitable_object::obj_type_str(object_type obj) {
    switch (obj) {
    case object_type::SPHERE:
        return "SPHERE";
    case object_type::MOVING_SPHERE:
        return "MOVING_SPHERE";
    case object_type::HITABLE_LIST:
        return "HITABLE_LIST";
    case object_type::BOUNDING_VOLUME_HIERARCHY:
        return "BOUNDING_VOLUME_HIERARCHY";
    case object_type::UNKNOWN:
        return "UNKNOWN";
    }
    return "";
}