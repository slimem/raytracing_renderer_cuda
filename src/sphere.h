#pragma once

#include "hitable_object.h"
#include "material.h"

class sphere : public hitable_object {
public:
    __device__ sphere() {};
    __device__ sphere(vec3 center, float radius, const material* mat)
        : _c(center), _r(radius), _m(mat) {};
    __device__ virtual bool hit(const ray& r, float tmin, float tmax, hit_record& hrec) const override;
    __device__ virtual bool bounding_box(float t0, float t1, AABB& box) const override;
    __device__ virtual ~sphere() noexcept override;

    __device__ virtual object_type get_object_type() const override {
        return object_type::SPHERE;
    }

private:
    vec3 _c;
    float _r = 0.f;
    const material* _m = nullptr;
};

class moving_sphere : public hitable_object {
public:
    __device__ moving_sphere() {};
    __device__ moving_sphere(vec3 center0, vec3 center1,
        float time0, float time1,
        float radius,
        const material* mat)
        : _c0(center0), _c1(center1),
          _t0(time0), _t1(time1),
          _r(radius),
          _m(mat) {}
    __device__ virtual bool hit(const ray& r, float tmin, float tmax, hit_record& hrec) const override;
    __device__ virtual bool bounding_box(float t0, float t1, AABB& box) const override;
    __device__ ~moving_sphere() noexcept override;

    __device__ virtual object_type get_object_type() const override {
        return object_type::MOVING_SPHERE;
    }

    __device__ inline vec3 center(float time) const {
        // interpolation between centers
        return _c0 + ((time - _t0) / (_t1 - _t0)) * (_c1 - _c0);
    }

private:
    vec3 _c0, _c1;
    float _t0, _t1;
    float _r;
    const material* _m = nullptr;
};

__device__ bool
sphere::hit(const ray& r, float tmin, float tmax, hit_record& hrec) const {
    vec3 oc = r.origin() - _c; // A - C
    // 1 - dot((p(​ t) - c)​ ,(p(​ t) - c​)) = R*R
    // 2 - dot((A​ + t*B ​- C)​ ,(A​ + t*B​ - C​)) = R*R (A is origin, B is direction)
    // 3 - t*t*dot(B,​ B)​ + 2*t*dot(B,A​-C​) + dot(A-C,A​-C​) - R*R = 0
    // we solve it as a 2nd degree polynomial with delta = b^2 - 4*a*c
    float a = vec3::dot(r.direction(), r.direction());
    float b = vec3::dot(oc, r.direction());
    float c = vec3::dot(oc, oc) - _r * _r;
    float delta = b * b - a * c;

    if (delta > 0) {
        // delta is strictly positive, we have two hit points
        // t is the parameter that gives the point of intersection
        // between the ray and the sphere
        // TODO: use __fsqrt_rz for sqrt and profile!!
        float t = (-b - sqrt(delta)) / a;
        if (t < tmax && t > tmin) {
            hrec.set_t(t);
            hrec.set_p(r.point_at_parameter(t));
            // Remember: sphere normal is intersection_p - center
            // to get a normalized vector, we devide by its magnitude which is
            // the sphere radius
            hrec.set_n((hrec.p() - _c) / _r);
            //printf(" --- CALCULATED NORMAL: %f,%f,%f\n", hrec.n().x(), hrec.n().y(), hrec.n().z());
            hrec.set_h(hit_object_type::SPHERE);

            hrec.set_m(_m);
            return true;
        }
        // we use the same variable
        t = (-b + sqrt(delta)) / a;
        if (t < tmax && t > tmin) {
            hrec.set_t(t);
            hrec.set_p(r.point_at_parameter(t));
            hrec.set_n((hrec.p() - _c) / _r);
            hrec.set_h(hit_object_type::SPHERE);
            hrec.set_m(_m);
            return true;
        }
    }
    
    return false;
}

__device__ bool
sphere::bounding_box(float t0, float t1, AABB& box) const {
    box = AABB(_c - vec3(_r), _c + vec3(_r));
    return true;
}

__device__
sphere::~sphere() noexcept {
    printf("Deleting sphere object at %p\n", this);
    if (_m) {
        printf("--Deleting material object at %p\n", _m);
        delete _m;
    }
}

__device__ bool
moving_sphere::hit(const ray& r, float tmin, float tmax, hit_record& hrec) const {
    vec3 oc = r.origin() - center(r.t());
    float a = vec3::dot(r.direction(), r.direction());
    float b = vec3::dot(oc, r.direction());
    float c = vec3::dot(oc, oc) - _r * _r;
    float delta = b * b - a * c;

    if (delta > 0) {
        float t = (-b - sqrt(delta)) / a;
        if (t < tmax && t > tmin) {
            hrec.set_t(t);
            hrec.set_p(r.point_at_parameter(t));
            hrec.set_n((hrec.p() - center(r.t())) / _r);

            hrec.set_h(hit_object_type::SPHERE);

            hrec.set_m(_m);
            return true;
        }
        // we use the same variable
        t = (-b + sqrt(delta)) / a;
        if (t < tmax && t > tmin) {
            hrec.set_t(t);
            hrec.set_p(r.point_at_parameter(t));
            hrec.set_n((hrec.p() - center(r.t())) / _r);
            hrec.set_h(hit_object_type::SPHERE);
            hrec.set_m(_m);
            return true;
        }
    }

    return false;
}

__device__ bool
moving_sphere::bounding_box(float t0, float t1, AABB& box) const {
    // box at t0
    AABB box0 = AABB(_c0 - vec3(_r), _c0 + vec3(_r));

    // box at t1
    AABB box1 = AABB(_c1 - vec3(_r), _c1 + vec3(_r));

    box = AABB::surrounding_box(box0, box1);
    return true;
}
__device__
moving_sphere::~moving_sphere() noexcept {
    printf("Deleting moving_sphere object at %p\n", this);
    if (_m) {
        printf("--Deleting material object at %p\n", _m);
        delete _m;
    }
}
