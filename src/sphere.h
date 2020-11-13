#pragma once

#include "hitable_object.h"

class sphere : public hitable_object {
public:
    __device__ sphere() {};
    __device__ sphere(vec3 center, float radius)
        : _c(center), _r(radius) {};
    __device__ virtual bool hit(const ray& r, float tmin, float tmax, hit_record& hrec) const override;
private:
    vec3 _c;
    float _r;
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
            return true;
        }
        // we use the same variable
        t = (-b + sqrt(delta)) / a;
        if (t < tmax && t > tmin) {
            hrec.set_t(t);
            hrec.set_p(r.point_at_parameter(t));
            hrec.set_n((hrec.p() - _c) / _r);
            return true;
        }
    }
    
    return false;
}