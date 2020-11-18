#pragma once

#include "ray.h"
#include "utils.h"

class AABB {
public:
    __device__ AABB() {
        _min = vec3(FLT_MAX);
        _max = vec3(FLT_MIN);
    }

    __device__ AABB(const vec3& a, const vec3& b)
        : _min(a), _max(b) {}

    __device__ inline const vec3& min() const { return _min; }
    __device__ inline const vec3& max() const { return _max; }

    __device__ bool hit(const ray& r, float tmin, float tmax) const;

    __device__ static const AABB surrounding_box(const AABB& box0, const AABB& box1);

private:
    vec3 _min, _max;
};

__device__ bool
AABB::hit(const ray& r, float tmin, float tmax) const {
    // ray position p in time is p(t) = A + tB
    // A is origin and B is direction (chech ray.h)
    // x0 = Ax + t0 * Bx => t0 = (x0 - Ax) / Bx
    // x1 = Ax + t1 * Bx => t1 = (x1 - Ax) / Bx
    
    /*for (uint8_t i = 0; i < 3; ++i) {
        float t0 = utils::fmin(
            (_min[i] - r.origin()[i]) / r.direction()[i],
            (_max[i] - r.origin()[i]) / r.direction()[i]
        );

        float t1 = utils::fmax(
            (_min[i] - r.origin()[i]) / r.direction()[i],
            (_max[i] - r.origin()[i]) / r.direction()[i]
        );

        tmin = fmax(t0, tmin);
        tmax = fmin(t1, tmax);
        if (tmax <= tmin) {
            return false;
        }
    }
    return true;*/
    // Andrew Kensler implementation (Pixar)
    // TODO: Use intinsics
    for (uint8_t i = 0; i < 3; ++i) {
        float invD = __fdiv_rz(1.f, r.direction()[i]);
        float t0 = (_min[i] - r.origin()[i]) * invD;
        float t1 = (_max[i] - r.origin()[i]) * invD;
        if (invD < 0.f) {
            utils::swap(t0, t1);
        }
        //min return a < b ? a : b;
        tmin = utils::fmax(t0, tmin);
        tmax = utils::fmin(t1, tmax);
        if (tmax <= tmin) {
            return false;
        }
    }
    return true;
}

__device__ const AABB
AABB::surrounding_box(const AABB& box0, const AABB& box1) {
    vec3 small(
        utils::fmin(box0.min().x(), box1.min().x()),
        utils::fmin(box0.min().y(), box1.min().y()),
        utils::fmin(box0.min().z(), box1.min().z())
    );

    vec3 big(
        utils::fmax(box0.max().x(), box1.max().x()),
        utils::fmax(box0.max().y(), box1.max().y()),
        utils::fmax(box0.max().z(), box1.max().z())
    );

    return AABB(small, big);
}